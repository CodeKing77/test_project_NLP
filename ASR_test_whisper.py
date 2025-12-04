###COODE DE TRANCRIPTION AUDIO AVEC WHISPER###

import whisper

model = whisper.load_model("base")
result = model.transcribe("audio_test.wav")
print(result["text"])
