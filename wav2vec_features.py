import torch
import torchaudio
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()

def extract_embedding(audio_path):

    try:
        speech, sr = librosa.load(audio_path, sr=16000, mono=True)
    except:
        return None

    input_values = processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

    # last_hidden_state: [1, time_steps, 768]
    hidden_states = outputs.last_hidden_state

    # MEAN POOLING  (this is the key fix)
    embedding = torch.mean(hidden_states, dim=1)

    return embedding.squeeze().cpu()