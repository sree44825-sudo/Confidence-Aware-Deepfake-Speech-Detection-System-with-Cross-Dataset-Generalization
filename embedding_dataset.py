import os
import torch
from torch.utils.data import Dataset
import random


class EmbeddingDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir examples:
        embeddings/train
        embeddings/val
        embeddings/test
        """

        self.files = []
        self.labels = []

        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")

        # ---- Load REAL embeddings ----
        if os.path.exists(real_dir):
            for file in os.listdir(real_dir):
                if file.endswith(".pt"):
                    self.files.append(os.path.join(real_dir, file))
                    self.labels.append(0)   # 0 = real

        # ---- Load FAKE embeddings ----
        if os.path.exists(fake_dir):
            for file in os.listdir(fake_dir):
                if file.endswith(".pt"):
                    self.files.append(os.path.join(fake_dir, file))
                    self.labels.append(1)   # 1 = fake

        print(f"Loaded {len(self.files)} embeddings from {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        try:
            # attempt to load embedding
            data = torch.load(path, map_location="cpu")

            # extract tensor from dictionary
            embedding = data["embedding"].float()

            # label tensor
            label = torch.tensor(self.labels[idx], dtype=torch.float32)

            return embedding, label

        except Exception:
            # Windows disk read sometimes fails randomly
            # instead of crashing training, resample another file
            new_idx = random.randint(0, len(self.files) - 1)
            return self.__getitem__(new_idx)
