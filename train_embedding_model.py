import torch
from torch.utils.data import DataLoader
from embedding_dataset import EmbeddingDataset
from embedding_model import EmbeddingClassifier

# ---- Prevent Windows CPU memory explosion ----
torch.set_num_threads(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Load datasets ----
train_dataset = EmbeddingDataset("embeddings/train")
val_dataset = EmbeddingDataset("embeddings/val")

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,        # VERY IMPORTANT (Windows fix)
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    num_workers=0,
    pin_memory=False
)


# ---- Model ----
model = EmbeddingClassifier().to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

EPOCHS = 30   # we will stop early automatically

best_acc = 0

# ---- Training Loop ----
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = (outputs > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)

    accuracy = correct / total

    print(f"\nEpoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.4f}")

    # ---- Save Best Model ----
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), "best_embedding_model.pth")
        print(">>> Best model saved!")

    # ---- Early stopping to avoid RAM crash ----
    if epoch >= 11:
        print("\nStopping early to prevent memory overflow.")
        break

print("\nTraining complete.")
print("Best training accuracy:", best_acc)
