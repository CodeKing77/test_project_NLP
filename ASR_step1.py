###CODE DE CHARGEMENT ET D'ANALYSE DE FICHIERS AUDIO AVEC LIBROSA###

import librosa
import soundfile as sf

audio_path = "audio_test.wav"

# Chargement de l'audio
y, sr = librosa.load(audio_path, sr=None)  # sr=None garde l'échantillonnage original

print("Durée:", librosa.get_duration(y=y, sr=sr), "secondes")
print("Taille du signal:", len(y))
print("Fréquence d'échantillonnage:", sr)
