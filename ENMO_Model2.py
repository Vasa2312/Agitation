import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import random
import numpy as np

# =========================================================
# --------- ECM (EFFICIENT CHANNEL MODULATION) MODULE
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

# =========================================================
# --------- CROSS-ATTENTION FUSION
# =========================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1) # Lowered dropout here

    def forward(self, query, key_value):
        query = query.unsqueeze(1)
        key_value = key_value.unsqueeze(1)
        attn_out, _ = self.attn(query, key_value, key_value)
        return self.norm(query.squeeze(1) + self.dropout(attn_out.squeeze(1)))

# =========================================================
# --------- AUGMENTATION ONLY (NO MIXUP)
# =========================================================
def augment_timeseries(x, noise_std=0.01, time_mask_ratio=0.05, channel_drop_prob=0.05):
    # REDUCED NOISE INTENSITY to make training easier
    B, T, C = x.shape
    x_aug = x.clone()
    
    if noise_std > 0:
        x_aug += noise_std * torch.randn_like(x_aug)
    
    mask_len = int(T * time_mask_ratio)
    if mask_len > 0:
        for b in range(B):
            start = random.randint(0, T - mask_len)
            x_aug[b, start:start + mask_len] = 0

    for c in range(C):
        if random.random() < channel_drop_prob:
            x_aug[:, :, c] = 0
            
    return x_aug

# =========================================================
# --------- FOCAL LOSS (NO SMOOTHING)
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)
        # CHANGE: Set smoothing to 0.0 to force strict predictions
        self.smoothing = smoothing 

    def forward(self, logits, targets):
        # CHANGE: No smoothing applied if self.smoothing is 0
        targets_smooth = targets.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        
        bce = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, pos_weight=self.pos_weight, reduction="none"
        )
        pt = torch.exp(-bce)
        focal_loss = ((1 - pt) ** self.gamma * bce).mean()
        return focal_loss

# =========================================================
# --------- MODEL COMPONENTS
# =========================================================
class TemporalPatchEmbed(nn.Module):
    def __init__(self, in_ch=6, embed_dim=64, kernel_size=5, stride=2, padding=2):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, embed_dim, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x.transpose(1, 2)).transpose(1, 2)
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
        x = x + self.pe[:, :T]
        cls = self.cls.expand(B, -1, -1)
        return torch.cat([cls, x], dim=1)

class AgitationHybridPro(nn.Module):
    def __init__(self, input_dim=6, embed_dim=128, lstm_hidden=64, num_heads=8, num_layers=4, conv_stride=2, max_len=1024, transformer_dropout=0.05, head_dropout=0.1, multisample_dropout=1):
        super().__init__()
        
        # 1. Feature Extraction
        self.patch = TemporalPatchEmbed(input_dim, embed_dim, stride=conv_stride)
        self.ecm = ECM(channels=embed_dim, kernel_size=3) 
        
        # 2. Sequence Modeling
        self.pos = LearnablePositionalEncoding(embed_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=transformer_dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=transformer_dropout)
        
        # 3. Fusion
        lstm_out_dim = lstm_hidden * 2
        assert lstm_out_dim == embed_dim, f"LSTM hidden*2 ({lstm_out_dim}) must match Embed Dim ({embed_dim})"
        
        self.fusion = CrossAttentionFusion(dim=embed_dim)

        # 4. Classification Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), 
            nn.GELU(), 
            nn.Dropout(head_dropout), 
            nn.Linear(128, 1)
        )
        self.ms_drop = multisample_dropout

    def forward(self, x):
        # Transformer Branch
        x_trans = self.patch(x)
        x_trans = self.ecm(x_trans) 
        x_trans = self.pos(x_trans)
        x_trans = self.encoder(x_trans)
        feat_trans = x_trans[:, 0]
        
        # LSTM Branch
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(x)
        feat_lstm = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # Cross Attention Fusion
        combined_feat = self.fusion(query=feat_trans, key_value=feat_lstm)
        
        return self.head(combined_feat).squeeze(-1)

# =========================================================
# --------- MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # 1. SETUP DUMMY DATA
    N_SAMPLES = 1000
    x_data = torch.randn(N_SAMPLES, 120, 6) 
    y_data = torch.cat([torch.zeros(900), torch.ones(100)]).long()
    
    idx = torch.randperm(N_SAMPLES)
    x_data = x_data[idx]
    y_data = y_data[idx]

    # 2. SAMPLER
    class_counts = [900, 100]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[t] for t in y_data]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # 3. MODEL - INCREASED CAPACITY
    # CHANGE: Increased embed_dim to 128 and lstm_hidden to 64
    model = AgitationHybridPro(
        embed_dim=128, 
        lstm_hidden=64,
        transformer_dropout=0.05, # CHANGE: Lower dropout
        head_dropout=0.1          # CHANGE: Lower dropout
    ).to(device)
    
    EPOCHS = 10
    LR = 3e-4
    
    # CHANGE: Lower weight decay
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5) 
    scheduler = OneCycleLR(optimizer, max_lr=LR, total_steps=EPOCHS * len(train_loader), pct_start=0.3)
    
    # CHANGE: No smoothing
    criterion = FocalLoss(gamma=2.5, pos_weight=torch.tensor([1.0], device=device), smoothing=0.0)

    # 4. TRAIN (NO MIXUP)
    print("Starting training (Optimization: Low Regularization to reduce Train Loss)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # CHANGE: Reduced Augmentation intensity
            x_aug = augment_timeseries(x_batch, noise_std=0.01, time_mask_ratio=0.05)
            
            # CHANGE: Removed MixUp. We feed x_aug directly.
            optimizer.zero_grad()
            logits = model(x_aug)
            
            loss = criterion(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
