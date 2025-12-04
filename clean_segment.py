###CODE DE NETTOYAGE ET NORMALISATION AUDIO AVEC NOISEREDUCE###

import os
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np

input_dir = "vad_segments"
output_dir = "clean_segments"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        file_path = os.path.join(input_dir, filename)

        # Load audio
        audio, sr = librosa.load(file_path, sr=16000, mono=True)

        # Noise reduction
        reduced_noise = nr.reduce_noise(y=audio, sr=sr)

        # Normalisation
        normalized = reduced_noise / np.max(np.abs(reduced_noise))

        # Save clean audio
        out_path = os.path.join(output_dir, filename)
        sf.write(out_path, normalized, sr)
        print(f"Nettoyé : {filename}")

print("\n✔ Nettoyage + normalisation terminés !")
