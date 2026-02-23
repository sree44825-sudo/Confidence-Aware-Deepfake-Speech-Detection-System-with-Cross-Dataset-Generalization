import os
import torch

bad_files = []

def scan(folder):
    print("Scanning:", folder)
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".pt"):
                path = os.path.join(root, f)
                try:
                    data = torch.load(path)

                    # verify correct structure
                    if "embedding" not in data:
                        bad_files.append(path)

                except Exception:
                    bad_files.append(path)

scan("embeddings/train")
scan("embeddings/val")

print("\nBAD FILES FOUND:", len(bad_files))

for f in bad_files:
    print(f)

with open("bad_embeddings.txt", "w") as file:
    for f in bad_files:
        file.write(f + "\n")

print("\nSaved to bad_embeddings.txt")
