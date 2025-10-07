from faster_whisper import WhisperModel, BatchedInferencePipeline, available_models, download_model
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from datetime import datetime
from time import perf_counter
import uvicorn
import os
import uuid
import sys

app = FastAPI()

MODEL_SIZE_ENV = os.getenv("MODEL_SIZE", "large")
DEVICE_ENV = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE_ENV = os.getenv("COMPUTE_TYPE", "float16")
OUTPUT_DIR_ENV = os.getenv("OUTPUT_DIR", "/rootPath/STT_Output")
BATCH_SIZE_ENV = os.getenv("BATCH_SIZE", None)

def test_available_models():
    models = available_models()
    assert isinstance(models, list)
    assert "tiny" in models

model = WhisperModel(MODEL_SIZE_ENV, device=DEVICE_ENV, compute_type=COMPUTE_TYPE_ENV)

@app.post("/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), model_name: str = Form("whisper-1")):
    filename = f"temp_{uuid.uuid4()}.wav"
    with open(filename, "wb") as f:
        f.write(await file.read())

    try:
        start_time = perf_counter()
        batch_size_val = int(BATCH_SIZE_ENV) if BATCH_SIZE_ENV is not None else None

        if batch_size_val is not None:
            print("Batch size:", batch_size_val)
            batched_model = BatchedInferencePipeline(model=model)
            segments_gen, info = batched_model.transcribe(
                filename,
                beam_size=5,
                batch_size=batch_size_val,
                log_progress=True,
                multilingual=True                 
            )
        else:
            print("Batching: No")
            segments_gen, info = model.transcribe(
                filename,
                beam_size=5,
                log_progress=True,
                multilingual=True 
            )
        text = "".join([segment.text for segment in segments_gen])
        segments = list(segments_gen)
        processing_time = perf_counter() - start_time
        
        header = (
            f"Used model      : {model}\n"
            f"Audio duration         : {info.duration:.2f} sec\n"
            f"Nr of char: {len(text)}\n"
            f"Treatment duration: {processing_time:.2f} sec\n"
            f"{'-'*40}\n\n"
        )

        output_dir = os.path.join(OUTPUT_DIR_ENV, "STT_output")
        os.makedirs(output_dir, exist_ok=True)

        current_datetime = datetime.now().strftime("%y%m%d_%H%M")
        original_basename = os.path.splitext(file.filename)[0]
        output_filename = f"{current_datetime}_{original_basename}.txt"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(text)

    finally:
        os.remove(filename)

    return JSONResponse(content={
        "text": text,
        "language": info.language
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9200)
