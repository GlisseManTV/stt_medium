from faster_whisper import WhisperModel
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from tqdm import tqdm
from datetime import datetime
from time import perf_counter
import uvicorn
import os
import uuid
import sys

app = FastAPI()

from faster_whisper import available_models, download_model


def test_available_models():
    models = available_models()
    assert isinstance(models, list)
    assert "tiny" in models
# Chargement du modèle rapide
model = WhisperModel("medium", device="cuda", compute_type="float16")

@app.post("/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), model_name: str = Form("whisper-1")):
    # Générer un nom temporaire pour le fichier audio
    filename = f"temp_{uuid.uuid4()}.wav"
    with open(filename, "wb") as f:
        f.write(await file.read())

    try:
        start_time = perf_counter()
        segments_gen, info = model.transcribe(filename, beam_size=5)
        segments = list(segments_gen)  # convertir le générateur en liste
        text = "".join([segment.text for segment in tqdm(segments, desc="Transcription", unit="segment", file=sys.stdout, ascii=True)])
        processing_time = perf_counter() - start_time
        
        header = (
            f"Modèle utilisé      : {model}\n"
            f"Durée audio         : {info.duration:.2f} secondes\n"
            f"Nombre de caractères: {len(text)}\n"
            f"Temps de traitement : {processing_time:.2f} secondes\n"
            f"{'-'*40}\n\n"
)

        # Créer le répertoire de sortie s'il n'existe pas
        output_dir = os.path.join(os.path.dirname(__file__), "STT_output")
        os.makedirs(output_dir, exist_ok=True)

        # Générer le nom du fichier de sortie
        current_datetime = datetime.now().strftime("%y%m%d_%H%M")
        original_basename = os.path.splitext(file.filename)[0]  # Nom du fichier sans extension
        output_filename = f"{current_datetime}_{original_basename}.txt"
        output_path = os.path.join(output_dir, output_filename)

        # Sauvegarder la transcription dans le fichier de sortie
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(text)

    finally:
        # Supprimer le fichier temporaire
        os.remove(filename)

    # Retourner la transcription en JSON
    return JSONResponse(content={
        "text": text,
        "language": info.language
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9200)