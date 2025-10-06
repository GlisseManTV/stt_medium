# Audio Transcription API with Whisper

This project provides a FastAPI-based service to transcribe audio files using the **Whisper** model from OpenAI, powered by the `faster-whisper` library for improved performance.

## 📌 Features
- Real-time transcription of audio files (WAV format)
- Supports multiple Whisper models (e.g., `tiny`, `medium`)
- GPU acceleration via CUDA and `float16` precision
- Output includes transcribed text and detected language
- Saves transcription results to a timestamped `.txt` file
- Lightweight and scalable for deployment

## 🛠️ Requirements
- Python 3.8+
- `fastapi`, `uvicorn`, `faster-whisper`, `tqdm`, `numpy`, `torch`, `torchvision`, `torchaudio`
- NVIDIA GPU with CUDA support (recommended for best performance)

## 📦 Installation
```bash
pip install fastapi uvicorn faster-whisper tqdm
```

## 🚀 Running the Server
1. Save the provided Python script as `main.py` (or similar).
2. Run the server:
   ```bash
   python main.py
   ```
3. The API will be available at `http://0.0.0.0:9200`

## 📤 API Endpoint
### POST `/audio/transcriptions`
Transcribes an uploaded audio file.

#### Parameters
| Parameter | Type | Required | Description |
|----------|------|----------|-------------|
| `file` | `File` | ✅ | Audio file in WAV format (max 256 MB) |
| `model_name` | `string` | ❌ | Model name (default: `whisper-1`, uses `medium` internally) |

#### Request Example (using curl)
```bash
curl -X POST http://0.0.0.0:9200/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model_name=medium"
```

#### Response (JSON)
```json
{
  "text": "Hello, this is a test audio file.",
  "language": "en"
}
```

## 📂 Output Files
Transcriptions are saved in the `STT_output` directory with the following naming pattern:
```
YYMMDD_HHMM_original_filename.txt
```
Example: `251006_1005_sample.txt`

The output includes:
- Model used
- Audio duration
- Character count
- Processing time
- Transcribed text

## ⚙️ Model Configuration
The current implementation uses the `medium` model with:
- `device="cuda"` → GPU acceleration
- `compute_type="float16"` → Mixed precision for speed

> 💡 Tip: For faster inference on CPU, use `tiny` or `base` models.

## 🧪 Testing
Run the test to verify available models:
```python
from faster_whisper import available_models
print(available_models())
```

## 📌 Notes
- The temporary file is automatically deleted after processing.
- The service does not store files permanently.
- Suitable for batch processing or real-time transcription in applications.

## 📄 License
MIT License

---

*Built with ❤️ using `faster-whisper` and `FastAPI`.*

[1] https://github.com/SYSTRAN/faster-whisper
[2] https://github.com/openai/whisper