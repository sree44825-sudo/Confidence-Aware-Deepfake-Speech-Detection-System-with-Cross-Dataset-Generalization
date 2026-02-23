import os
import torch
import librosa
from embedding_model import EmbeddingClassifier
from wav2vec_features import extract_embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load trained classifier
model = EmbeddingClassifier().to(device)
model.load_state_dict(torch.load("best_embedding_model.pth", map_location=device))
model.eval()

fake_folder = "wavefake_test/fake"

total = 0
detected_fake = 0

files = os.listdir(fake_folder)

for i, file in enumerate(files):

    if i % 50 == 0:
        print("Processed", i, "/", len(files))

    if not file.endswith(".wav"):
        continue

    path = os.path.join(fake_folder, file)

    try:
        embedding = extract_embedding(path)
        embedding = embedding.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(embedding)
            prediction = (output > 0.5).item()

        total += 1

        # 1 = fake prediction
        if prediction == 1:
            detected_fake += 1

    except Exception:
        continue

print("\n------ WaveFake Cross Dataset Result ------")
print("Total fake audios:", total)
print("Detected as fake:", detected_fake)
print("Missed:", total - detected_fake)
print("Detection Rate:", detected_fake / total)
