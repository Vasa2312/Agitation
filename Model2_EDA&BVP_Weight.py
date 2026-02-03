import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from tqdm import tqdm
import random

# =========================================================
# 1. MIXUP HELPER FUNCTIONS
# =========================================================
def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, priority_a=None, priority_b=None):
    '''
    Calculates weighted loss for mixed targets.
    Supports priority weights for both mixed samples.
    '''
    # Calculate loss for first set of targets
    loss_a = criterion(pred, y_a, priority_weights=priority_a)
    
    # Calculate loss for second set of targets
    loss_b = criterion(pred, y_b, priority_weights=priority_b)
    
    return lam * loss_a + (1 - lam) * loss_b

# =========================================================
# 2. PRIORITY FOCAL LOSS
# =========================================================
class PriorityFocalLoss(nn.Module):
    def __init__(self, gamma=2.5, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets, priority_weights=None):
        """
        Calculates Focal Loss and scales it by priority_weights.
        """
        if targets.dim() > 1: 
            targets = targets.squeeze()
        if logits.dim() > 1: 
            logits = logits.squeeze()
            
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight, reduction="none"
        )
        
        pt = torch.exp(-bce)
        focal_loss = ((1 - pt) ** self.gamma * bce)
        
        # Apply Priority Weights (BVP Boosting)
        if priority_weights is not None:
            if priority_weights.dim() > 1:
                priority_weights = priority_weights.squeeze()
            focal_loss = focal_loss * priority_weights
            
        return focal_loss.mean()

# =========================================================
# 3. HELPER TO CALCULATE BVP PRIORITIES
# =========================================================
def get_bvp_priorities(x_batch, bvp_channel_idx=1, threshold=700, boost=5.0):
    """
    Returns a tensor of weights: boost if mean BVP < threshold, else 1.0
    """
    # x_batch shape: [Batch, Time, Channels]
    bvp_signal = x_batch[:, :, bvp_channel_idx]
    
    # Compute mean BVP per sample
    mean_bvp = bvp_signal.mean(dim=1)
    
    # Boost weight if mean BVP < threshold
    bvp_mask = (mean_bvp < threshold).float()
    priorities = 1.0 + bvp_mask * (boost - 1.0)
    
    return priorities.to(x_batch.device)

# =========================================================
# 4. MODEL DEFINITION
# =========================================================
class ECM(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.avg_pool(y)
        y = y.transpose(1, 2)
        y = self.conv(y)
        y = self.sigmoid(y)   
        return x * y

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1) 
    def forward(self, query, key_value):
        query = query.unsqueeze(1)
        key_value = key_value.unsqueeze(1)
        attn_out, _ = self.attn(query, key_value, key_value)
        return self.norm(query.squeeze(1) + self.dropout(attn_out.squeeze(1)))

class TemporalPatchEmbed(nn.Module):
    def __init__(self, in_ch=4, embed_dim=64, kernel_size=5, stride=2, padding=2):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, embed_dim, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return self.norm(x)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
    def forward(self, x):
        B, T, _ = x.shape
        if T > self.pe.shape[1]: 
            x = x[:, :self.pe.shape[1], :]
            T = x.shape[1]
        x = x + self.pe[:, :T]
        cls = self.cls.expand(B, -1, -1)
        return torch.cat([cls, x], dim=1)

class AgitationHybridPro(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128, lstm_hidden=64, num_heads=8, num_layers=4, 
                 conv_stride=2, max_len=1024, transformer_dropout=0.05, head_dropout=0.1):
        super().__init__()
        self.patch = TemporalPatchEmbed(input_dim, embed_dim, stride=conv_stride)
        self.ecm = ECM(channels=embed_dim, kernel_size=3) 
        self.pos = LearnablePositionalEncoding(embed_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                               dim_feedforward=embed_dim * 4, 
                                               dropout=transformer_dropout, 
                                               activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=transformer_dropout)
        self.fusion = CrossAttentionFusion(dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), 
            nn.GELU(), 
            nn.Dropout(head_dropout), 
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_trans = self.patch(x)
        x_trans = self.ecm(x_trans)
        x_trans = self.pos(x_trans)
        x_trans = self.encoder(x_trans)
        feat_trans = x_trans[:, 0]

        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(x)
        feat_lstm = torch.cat([h_n[-2], h_n[-1]], dim=1)

        combined_feat = self.fusion(query=feat_trans, key_value=feat_lstm)
        return self.head(combined_feat).squeeze(-1)

# =========================================================
# 5. DATASET CLASS
# =========================================================
class AgitationDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, downsample=1, fit_scaler=False, scaler=None):
        self.data = data[:, ::downsample, :]
        self.labels = torch.tensor(labels, dtype=torch.float32)

        B, T, C = self.data.shape
        flat_data = self.data.reshape(-1, C)

        if fit_scaler:
            self.scaler = {
                'mean': np.mean(flat_data, axis=0),
                'std': np.std(flat_data, axis=0) + 1e-8
            }
        elif scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = {'mean': 0, 'std': 1}

        flat_data = (flat_data - self.scaler['mean']) / self.scaler['std']
        self.data = torch.tensor(flat_data.reshape(B, T, C), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# =========================================================
# 6. TRAINING FUNCTION (example snippet)
# =========================================================
def train_and_evaluate_fold(fold_id, train_indices, test_indices, full_data, full_labels, 
                            downsample, epochs, batch_size, accum_steps):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, train_labels = full_data[train_indices], full_labels[train_indices]
    test_data, test_labels = full_data[test_indices], full_labels[test_indices]

    train_dataset = AgitationDataset(train_data, train_labels, downsample=downsample, fit_scaler=True)
    test_dataset = AgitationDataset(test_data, test_labels, downsample=downsample, scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AgitationHybridPro(input_dim=full_data.shape[2]).to(device)
    criterion = PriorityFocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # --- Priority weights based on BVP < 700 ---
            priority_weights = get_bvp_priorities(x_batch, bvp_channel_idx=1, threshold=700, boost=5.0)

            # Forward pass
            pred = model(x_batch)
            loss = criterion(pred, y_batch, priority_weights=priority_weights)
            loss.backward()
            optimizer.step()

        print(f"Fold {fold_id}, Epoch {epoch+1}, Loss: {loss.item():.4f}")

