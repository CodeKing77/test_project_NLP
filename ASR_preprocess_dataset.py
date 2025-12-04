###CODE DE PREPROCESSING DE DATASET AUDIO POUR ASR AVEC WHISPER###

import os
import soundfile as sf
from pydub import AudioSegment, silence
import whisper
import pandas as pd



# 1️⃣ Configuration
AUDIO_FILE = "audio_CIV.wav"  # Ton fichier audio
OUTPUT_DIR = "dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2️⃣ Charger modèle Whisper
model = whisper.load_model("small")  # tu peux mettre "small" si ton PC est costaud -  small, base, tiny, etc.

# 3️⃣ Charger l'audio
audio = AudioSegment.from_wav(AUDIO_FILE)

# Détection des silences : segments = parole détectée
chunks = silence.split_on_silence(audio,
    min_silence_len=700,  # silence min 0.7s
    silence_thresh=-40    # seuil de détection
)

metadata = []

print(f"Nombre de segments trouvés : {len(chunks)}")

for i, chunk in enumerate(chunks):
    file_name = f"sample_{i}.wav"
    file_path = os.path.join(OUTPUT_DIR, file_name)

    chunk.export(file_path, format="wav")  # Sauvegarde du segment

    # 4️⃣ Transcription du segment
    result = model.transcribe(file_path, language="fr")
    transcript = result["text"].strip()

    metadata.append([file_name, transcript])
    print(f"{file_name} -> {transcript}")

# 5️⃣ Sauvegarde du dataset
df = pd.DataFrame(metadata, columns=["file_name", "transcript"])
df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

print("\n📦 Dataset créé avec succès ! 🎯")