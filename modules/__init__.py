from fastapi import APIRouter

from .vector import api as vector
from .ocr import api as ocr
from .summary import api as summary

router = APIRouter()

router.include_router(vector.router, prefix="/vector")
router.include_router(ocr.router, prefix="/ocr")
router.include_router(summary.router, prefix="/summary")
