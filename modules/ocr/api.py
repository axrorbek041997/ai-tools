import io

import easyocr
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()

# ocr
ocr_reader = easyocr.Reader(["en"], gpu=False)  # add languages like ["en","ru","uz"] if you need

def _to_py(v):
    """Convert numpy scalars/arrays to pure Python types for JSON."""
    if isinstance(v, np.generic):          # numpy scalar (int32, float32, ...)
        return v.item()
    if isinstance(v, np.ndarray):          # numpy array
        return v.tolist()
    if isinstance(v, (list, tuple)):       # list/tuple possibly containing numpy types
        return [_to_py(x) for x in v]
    return v

@router.post("/")
async def ocr_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    img_np = np.array(img)

    # results: [ [bbox, text, conf], ... ]
    results = ocr_reader.readtext(img_np)

    items = []
    texts = []
    for bbox, text, conf in results:
        texts.append(text)
        items.append({
            "text": str(text),
            "confidence": float(conf),
            # bbox is 4 points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            "bbox": _to_py(bbox),
        })

    return {
        "items": items,
        "text": " ".join(texts).strip(),
    }
