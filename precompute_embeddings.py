import os
import torch
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

device = "cpu"

print("Loading Wav2Vec2...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()

def process_folder(input_dir, output_dir, label):
    os.makedirs(output_dir, exist_ok=True)

    files = []
    for root, _, filenames in os.walk(input_dir):
        for f in filenames:
            if f.endswith(".wav") or f.endswith(".mp3"):
                files.append(os.path.join(root, f))

    print(f"\nProcessing {len(files)} files from {input_dir}")

    for path in tqdm(files):
        try:
            speech, sr = librosa.load(path, sr=16000, mono=True)

            inputs = processor(speech, sampling_rate=16000, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

            name = os.path.basename(path).split(".")[0]
            save_path = os.path.join(output_dir, name + ".pt")

            torch.save({
                "embedding": embedding,
                "label": label
            }, save_path)

        except:
            print("Skipped:", path)

# TRAIN
process_folder("data/train/real", "embeddings/train/real", 0)
process_folder("data/train/fake", "embeddings/train/fake", 1)

# VAL
process_folder("data/val/real", "embeddings/val/real", 0)
process_folder("data/val/fake", "embeddings/val/fake", 1)

# TEST
process_folder("data/test/real", "embeddings/test/real", 0)
process_folder("data/test/fake", "embeddings/test/fake", 1)

print("DONE — embeddings saved.")