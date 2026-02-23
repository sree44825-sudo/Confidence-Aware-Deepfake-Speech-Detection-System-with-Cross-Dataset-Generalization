import os
import librosa

bad_files = []

def scan_folder(folder):
    print("Scanning:", folder)

    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".wav") or file.endswith(".mp3"):
                path = os.path.join(root, file)

                try:
                    librosa.load(path, sr=16000)
                except:
                    print("Corrupted:", path)
                    bad_files.append(path)

scan_folder("data/train")
scan_folder("data/val")
scan_folder("data/test")

print("\nTotal corrupted files:", len(bad_files))

# save list
with open("bad_files.txt", "w") as f:
    for file in bad_files:
        f.write(file + "\n")

print("Saved list to bad_files.txt")