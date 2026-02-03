import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from dataset import AgitationDataset
from model_2 import AgitationHybridPro 

# ------------------- Utils -------------------

def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg["lr"]

def maybe_log_lr_change(prev_lr, optimizer):
    curr_lr = get_lr(optimizer)
    if curr_lr < prev_lr:
        print(f"[Scheduler] Learning rate reduced: {prev_lr:.6f} -> {curr_lr:.6f}")
    return curr_lr

# ----------------- BVP PRIORITY LOSS ------------------

class PriorityFocalLoss(nn.Module):
    def __init__(self, gamma=2.5, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets, priority_weights=None):
        if targets.dim() > 1: 
            targets = targets.squeeze()
        if logits.dim() > 1: 
            logits = logits.squeeze()

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight, reduction="none"
        )
        pt = torch.exp(-bce)
        focal_loss = ((1 - pt) ** self.gamma * bce)

        if priority_weights is not None:
            if priority_weights.dim() > 1:
                priority_weights = priority_weights.squeeze()
            focal_loss = focal_loss * priority_weights

        return focal_loss.mean()

def get_bvp_priorities(x_batch, bvp_channel_idx=1, threshold=700, boost=5.0):
    """
    Returns priority weights: boost if mean BVP < threshold, else 1.0
    """
    bvp_signal = x_batch[:, :, bvp_channel_idx]
    mean_bvp = bvp_signal.mean(dim=1)
    mask = (mean_bvp < threshold).float()
    priorities = 1.0 + mask * (boost - 1.0)
    return priorities.to(x_batch.device)

# --------------- Train/Eval for one fold ---------------

def train_and_evaluate_fold(
    fold_id: int,
    train_indices,
    test_indices,
    full_data,
    full_labels,
    downsample: int,
    epochs: int,
    batch_size: int,
    num_workers: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,
    early_patience: int = 10,
    accum_steps: int = 1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split arrays
    train_data = full_data[train_indices]
    train_labels = full_labels[train_indices]
    test_data  = full_data[test_indices]
    test_labels = full_labels[test_indices]

    # Datasets
    train_ds = AgitationDataset(train_data, train_labels, downsample, fit_scaler=True)
    test_ds  = AgitationDataset(test_data,  test_labels,  downsample, scaler=train_ds.scaler)

    # Imbalance Fix (WeightedRandomSampler)
    n_neg = (train_labels == 0).sum()
    n_pos = (train_labels == 1).sum()
    n_neg = max(n_neg, 1)
    n_pos = max(n_pos, 1)

    weight_neg = 1.0 / n_neg
    weight_pos = 1.0 / n_pos
    
    samples_weights = torch.tensor([
        weight_pos if label == 1 else weight_neg 
        for label in train_labels
    ], dtype=torch.double)
    
    sampler = WeightedRandomSampler(
        weights=samples_weights, 
        num_samples=len(samples_weights), 
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,      
        sampler=sampler,    
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader  = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    criterion = PriorityFocalLoss()  # <-- Use BVP-based priority loss

    # Model
    model = AgitationHybridPro(input_dim=full_data.shape[2]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_auprc = 0.0  
    best_path = f"best_pro_fold{fold_id}.pt"
    no_improve = 0
    prev_lr = get_lr(optimizer)
    accum_steps = max(1, accum_steps)

    for epoch in range(1, epochs + 1):
        # --------- Train ---------
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"[Fold {fold_id}] Train E{epoch}/{epochs}")
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)

            logits = model(x)

            # --- BVP priority weights ---
            priority_weights = get_bvp_priorities(x, bvp_channel_idx=1, threshold=700, boost=5.0)

            loss = criterion(logits, y, priority_weights=priority_weights)
            running_loss += loss.item()

            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, len(train_loader))

        # --------- Eval ---------
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"[Fold {fold_id}] Eval  E{epoch}/{epochs}"):
                x = x.to(device, non_blocking=True)
                logits = model(x)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y.numpy())

        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)

        print(f"[Fold {fold_id}] Epoch {epoch}: train_loss={train_loss:.4f} AUC={auc:.4f} AUPRC={auprc:.4f}")

        # --------- Optimize ---------
        scheduler.step(auprc)
        prev_lr = maybe_log_lr_change(prev_lr, optimizer)

        if auprc > best_auprc:
            best_auprc = auprc
            torch.save(model.state_dict(), best_path)
            no_improve = 0
            print(f"[Fold {fold_id}] ✅ New best AUPRC {best_auprc:.4f}")
        else:
            no_improve += 1
            if no_improve >= early_patience:
                print(f"[Fold {fold_id}] Early stopping.")
                break

    print(f"[Fold {fold_id}] Finished. Best Test AUPRC: {best_auprc:.4f}")
    return best_auprc


# ------------------- Main -------------------

if __name__ == "__main__":
    seed_everything(42)

    # 1. LOAD DATA
    print("Loading data...")
    try:
        X = np.load("processed_17p/X_train.npy", mmap_mode='r')
        y = np.load("processed_17p/y_train.npy", mmap_mode='r')
        print(f"Loaded: X={X.shape}  y={y.shape}")
    except FileNotFoundError:
        print("❌ Error: Could not find processed_17p/X_train.npy. Run preprocessing first.")
        exit()

    DOWNSAMPLE_FACTOR = 16      
    MAX_EPOCHS = 50
    BATCH_SIZE = 16             
    TARGET_BATCH = 512          
    ACCUM_STEPS = max(1, TARGET_BATCH // BATCH_SIZE)   

    print(f"\nConfig: Batch={BATCH_SIZE} | Accum={ACCUM_STEPS} | Epochs={MAX_EPOCHS}")

    # Standard KFold
    N_FOLDS = 2
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    all_test_auprcs = []
    
    for fold_id, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
        print(f"\n================== Starting Fold {fold_id}/{N_FOLDS} ==================")
        
        fold_auprc = train_and_evaluate_fold(
            fold_id=fold_id,
            train_indices=train_index,
            test_indices=test_index,
            full_data=X,
            full_labels=y,
            downsample=DOWNSAMPLE_FACTOR,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            num_workers=0, 
            lr=1e-3,
            weight_decay=1e-3,
            grad_clip=1.0,
            early_patience=4,
            accum_steps=ACCUM_STEPS,
        )
        all_test_auprcs.append(fold_auprc)

    avg_auprc = float(np.mean(all_test_auprcs))
    std_auprc = float(np.std(all_test_auprcs))
    print("\n========================= Training Complete =========================")
    print(f"Final Report (Mixed Patient Split) — Avg AUPRC: {avg_auprc:.4f} ± {std_auprc:.4f}")
