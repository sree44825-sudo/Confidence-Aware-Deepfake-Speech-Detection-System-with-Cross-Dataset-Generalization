import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

print("Loading wav2vec model... (first time will download ~360MB)")

# load pretrained speech model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

print("Model loaded successfully!")

# pick ANY one audio file from your dataset
audio_path = "data_wavefake/test/fake/LJ012-0028_generated.wav"
  # <-- change if name different

print("Loading audio:", audio_path)

speech, sr = librosa.load(audio_path, sr=16000)

inputs = processor(speech, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state

print("Embedding shape:", embeddings.shape)
print("SUCCESS — wav2vec feature extraction works")
