import os
import torch
import librosa
from torch.utils.data import Dataset
from wav2vec_features import extract_embedding

class Wav2VecAudioDataset(Dataset):
    def __init__(self, root_dir):
        self.audio_paths = []
        self.labels = []

        for label_name, label_value in [("real", 0), ("fake", 1)]:
            class_path = os.path.join(root_dir, label_name)

            if not os.path.exists(class_path):
                continue

            for file in os.listdir(class_path):
                if file.endswith(".wav") or file.endswith(".mp3"):
                    self.audio_paths.append(os.path.join(class_path, file))
                    self.labels.append(label_value)

        print(f"Loaded {len(self.audio_paths)} files from {root_dir}")

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        embedding = extract_embedding(path)

        return embedding, label