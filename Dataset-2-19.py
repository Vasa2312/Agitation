import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

class AgitationDataset(Dataset):
    def __init__(self, data, labels, patient_ids, downsample=1, scaler=None, fit_scaler=False):
        """
        Args:
            data: np.array [N, Time, Channels]
            labels: np.array [N]
            patient_ids: np.array [N]
            downsample: int (Factor to reduce sequence length)
            scaler: sklearn scaler instance
            fit_scaler: bool (Whether to fit a new scaler on this data)
        """
        self.x = data[:, ::downsample, :]
        self.y = labels
        self.pids = patient_ids  # Store Patient IDs

        # Scaling logic
        if fit_scaler:
            self.scaler = StandardScaler()
            # Flatten to [N * Time, Channels] to fit
            flat = self.x.reshape(-1, self.x.shape[-1])
            self.scaler.fit(flat)
        else:
            self.scaler = scaler

        if self.scaler is not None:
            shape = self.x.shape
            flat = self.x.reshape(-1, shape[-1])
            flat = self.scaler.transform(flat)
            self.x = flat.reshape(shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return signals, labels, AND patient_id
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.pids[idx], dtype=torch.long)
        )
