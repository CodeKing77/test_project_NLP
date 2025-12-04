###CODE DE SEGMENTATION VAD (VOICE ACTIVITY DETECTION) AVEC WEBRTC VAD###

import os
import librosa
import numpy as np
import soundfile as sf
import webrtcvad

audio_path = "audio_CIV.wav"  # Chemin vers mon fichier audio
output_dir = "vad_segments"
os.makedirs(output_dir, exist_ok=True)

# 1) Charger l’audio en mono 16kHz
audio, sr = librosa.load(audio_path, sr=16000)
print(f"Audio chargé | durée = {len(audio)/sr:.2f}s | sr={sr}")

# 2) Config VAD : 0 (tolérant) → 3 (strict)
vad = webrtcvad.Vad(2)

frame_duration = 30  # 30ms
frame_len = int(sr * frame_duration / 1000)

segments = []
current_segment = []

# 3) Analyse frame par frame
for i in range(0, len(audio), frame_len):
    frame = audio[i:i+frame_len]
    if len(frame) < frame_len: break

    pcm = (frame * 32768).astype(np.int16).tobytes()

    if vad.is_speech(pcm, sr):
        current_segment.extend(frame)
    else:
        if len(current_segment) > 0:
            segments.append(np.array(current_segment))
            current_segment = []

# Dernier segment si pas terminé
if len(current_segment) > 0:
    segments.append(np.array(current_segment))

# 4) Sauvegarde des segments vocaux
for i, seg in enumerate(segments):
    out_path = os.path.join(output_dir, f"seg_{i}.wav")
    sf.write(out_path, seg, sr)

print(f"Segments vocaux sauvegardés : {len(segments)} fichiers")
