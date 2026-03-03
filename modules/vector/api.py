import io

import numpy as np
import torch
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from scipy.io import wavfile
from scipy.signal import resample_poly
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from transformers import pipeline

router = APIRouter()

# Load models once (on startup)
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)

asr_model_name = "openai/whisper-tiny"
asr = pipeline(
    task="automatic-speech-recognition",
    model=asr_model_name,
    device=0 if device == "cuda" else -1,
)

class TextIn(BaseModel):
    text: str


def _decode_wav_bytes(raw: bytes, target_sr: int = 16000) -> np.ndarray:
    try:
        sample_rate, audio = wavfile.read(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid WAV audio")

    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Audio is empty")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_val = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / float(max_val)
    else:
        audio = audio.astype(np.float32)

    if sample_rate != target_sr:
        audio = resample_poly(audio, target_sr, sample_rate).astype(np.float32)

    return np.ascontiguousarray(audio)

@router.post("/text")
def vectorize_text(payload: TextIn):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    vec = text_model.encode(payload.text, normalize_embeddings=True)
    return {
        "type": "text",
        "dim": int(vec.shape[0]),
        "vector": vec.tolist(),
    }


@router.post("/image")
async def vectorize_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # IMPORTANT: make sure processor puts tensors on CPU first, then move to device
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = clip_model.get_image_features(**inputs)

        # Some envs / wrappers may return an output object instead of a tensor
        if not torch.is_tensor(out):
            if hasattr(out, "image_embeds") and out.image_embeds is not None:
                out = out.image_embeds
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                out = out.pooler_output
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected CLIP output type: {type(out)}"
                )

        img_features = torch.nn.functional.normalize(out, p=2, dim=-1)

    vec = img_features[0].detach().cpu().numpy()
    return {
        "type": "image",
        "dim": int(vec.shape[0]),
        "vector": vec.tolist(),
    }


@router.post("/audio")
async def audio_to_text(
    file: UploadFile = File(...),
    language: str = Query(default="en", min_length=2, max_length=10),
):
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Please upload an audio file")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Audio is empty")

    normalized_language = language.strip().lower()
    if not normalized_language:
        raise HTTPException(status_code=400, detail="language is empty")

    decode_kwargs = {"language": normalized_language, "task": "transcribe"}

    # Fast path for WAV; fallback lets transformers/ffmpeg decode MP3/M4A/etc.
    try:
        audio = _decode_wav_bytes(raw, target_sr=16000)
        result = asr(
            {"array": audio, "sampling_rate": 16000},
            return_timestamps=True,
            generate_kwargs=decode_kwargs,
        )
    except HTTPException:
        try:
            result = asr(raw, return_timestamps=True, generate_kwargs=decode_kwargs)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unsupported audio format: {exc}")

    text = str(result.get("text", "")).strip()

    return {
        "type": "audio",
        "language": normalized_language,
        "text": text,
    }
