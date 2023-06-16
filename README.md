# whisper-tflite

## Getting Started
- Download the whisper-tiny-en tflite model from [here](https://drive.google.com/file/d/1-YRvSVbAk-hXGNXMK6wMm04eqULSPg3Q/view?usp=sharing)

- Install requirements
```bash
 pip install -r requirements.txt
```

- Transcribe with tflite model
```python

 from whisper_tflite import WhisperModel
  
 model = WhisperModel("./whisper-tiny-en.tflite")

 segments, _ = model.transcribe("audio.mp3")
 
 for segment in segments:
     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
 
```

## References
Most of the code is taken from [faster-whisper](https://github.com/guillaumekln/faster-whisper/)