###CODE D'EXTRACTION DE CARACTERISTIQUES AUDIO AVEC LIBROSA###

import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_file = "audio_test.wav"
y, sr = librosa.load(audio_file)

# MFCC --Extraction des MFCC (utiles pour la reconnaissance vocale)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print("MFCC shape:", mfcc.shape)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title("MFCC")
plt.show()

#Détection des silences
intervals = librosa.effects.split(y, top_db=30)
print("Silences détectés :", intervals)