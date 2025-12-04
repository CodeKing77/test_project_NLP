###CODE D'ANALYSE ET VISUALISATION DE SPECTROGRAMME AVEC LIBROSA###

import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_file = "audio_test.wav"
y, sr = librosa.load(audio_file)

# Spectrogramme
S = librosa.stft(y)
S_db = librosa.amplitude_to_db(abs(S))

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title("Spectrogramme")
plt.show()







