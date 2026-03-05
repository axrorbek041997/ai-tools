import os
import re
from typing import Dict, List, Optional

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

router = APIRouter()


class SummarizeIn(BaseModel):
    text: str = Field(..., description="Large input text, e.g. a full news article")
    language: str = Field(default="en", min_length=2, max_length=10)
    min_length: int = Field(default=80, ge=20, le=400)
    max_length: int = Field(default=220, ge=50, le=800)


class NewsSummarizeIn(SummarizeIn):
    title: str = Field(default="", description="Optional news headline")
    impact_mode: str = Field(
        default="all",
        description="Choose one: all | industry | sector | stock",
    )
    industry_keywords: List[str] = Field(
        default_factory=list,
        description="Words/phrases representing an industry focus",
    )
    sector_keywords: List[str] = Field(
        default_factory=list,
        description="Words/phrases representing a sector focus",
    )
    stocks: List[str] = Field(
        default_factory=list,
        description="Stock tickers to assess (e.g., AAPL, TSLA). If empty, auto-detect from text.",
    )


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_into_chunks(text: str, max_words: int = 700, overlap_words: int = 70) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def _cheap_extractive_summary(text: str, target_sentences: int = 5) -> str:
    # Fallback path when model inference is unavailable.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return ""
    return " ".join(sentences[:target_sentences])


def _lexicon_sentiment(text: str) -> Dict[str, float | str]:
    # Lightweight fallback when sentiment model is unavailable.
    positive_words = {
        "good", "great", "positive", "growth", "improve", "improved", "improvement",
        "strong", "success", "benefit", "win", "record", "surge", "gain", "optimistic",
    }
    negative_words = {
        "bad", "worse", "negative", "decline", "drop", "loss", "risk", "crisis",
        "fail", "failure", "weak", "concern", "concerns", "inflation", "recession",
    }
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return {"label": "neutral", "score": 0.0}

    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    total = pos + neg

    if total == 0:
        return {"label": "neutral", "score": 0.0}
    if pos > neg:
        return {"label": "positive", "score": round((pos - neg) / total, 4)}
    if neg > pos:
        return {"label": "negative", "score": round((neg - pos) / total, 4)}
    return {"label": "neutral", "score": 0.0}


def _extract_tickers(text: str) -> List[str]:
    # Naive ticker extraction for uppercase symbols.
    blacklist = {
        "USA", "US", "EU", "UK", "CEO", "CFO", "GDP", "SEC", "ETF", "NYSE", "NASDAQ",
        "AI", "IPO", "USD", "EPS", "Q1", "Q2", "Q3", "Q4",
    }
    found = re.findall(r"\b[A-Z]{1,5}\b", text)
    uniq: List[str] = []
    for token in found:
        if token in blacklist:
            continue
        if token not in uniq:
            uniq.append(token)
    return uniq[:10]


def _filter_text_by_keywords(text: str, keywords: List[str]) -> str:
    if not keywords:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    lowered = [k.strip().lower() for k in keywords if k.strip()]
    selected = []
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        sl = s.lower()
        if any(k in sl for k in lowered):
            selected.append(s)
    return _normalize_whitespace(" ".join(selected))


device = "cuda" if torch.cuda.is_available() else "cpu"
summary_model_name = os.getenv("SUMMARY_MODEL_NAME", "facebook/bart-large-cnn")
sentiment_model_name = os.getenv("SENTIMENT_MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
summary_local_only = os.getenv("SUMMARY_LOCAL_FILES_ONLY", "0").strip().lower() in {"1", "true", "yes"}
_summarizer = None
_summarizer_error: Optional[str] = None
_sentiment = None
_sentiment_error: Optional[str] = None


def _get_summarizer():
    global _summarizer, _summarizer_error
    if _summarizer is not None:
        return _summarizer
    if _summarizer_error is not None:
        return None

    try:
        _summarizer = pipeline(
            task="summarization",
            model=summary_model_name,
            device=0 if device == "cuda" else -1,
            local_files_only=summary_local_only,
        )
        return _summarizer
    except Exception as exc:
        _summarizer_error = str(exc)
        return None


def _get_sentiment():
    global _sentiment, _sentiment_error
    if _sentiment is not None:
        return _sentiment
    if _sentiment_error is not None:
        return None

    try:
        _sentiment = pipeline(
            task="sentiment-analysis",
            model=sentiment_model_name,
            device=0 if device == "cuda" else -1,
            local_files_only=summary_local_only,
        )
        return _sentiment
    except Exception as exc:
        _sentiment_error = str(exc)
        return None


def _summarize_large_text(
    text: str,
    min_length: int,
    max_length: int,
) -> Dict[str, int | bool | str]:
    chunks = _split_into_chunks(text)
    per_chunk_min = max(20, min(min_length, max_length // 2))
    per_chunk_max = max(per_chunk_min + 10, min(max_length, 180))
    summarizer = _get_summarizer()

    partial_summaries: List[str] = []
    for chunk in chunks:
        if summarizer is None:
            partial_summaries.append(_cheap_extractive_summary(chunk, target_sentences=3))
            continue
        try:
            out = summarizer(
                chunk,
                min_length=per_chunk_min,
                max_length=per_chunk_max,
                do_sample=False,
            )
            partial_summaries.append(out[0]["summary_text"].strip())
        except Exception:
            partial_summaries.append(_cheap_extractive_summary(chunk, target_sentences=3))

    merged = _normalize_whitespace(" ".join(partial_summaries))
    if len(partial_summaries) > 1 and summarizer is not None:
        try:
            final = summarizer(
                merged,
                min_length=min_length,
                max_length=max_length,
                do_sample=False,
            )[0]["summary_text"].strip()
        except Exception:
            final = _cheap_extractive_summary(merged, target_sentences=6)
    else:
        final = merged if len(partial_summaries) == 1 else _cheap_extractive_summary(merged, target_sentences=6)

    return {
        "summary": final,
        "chunks": len(chunks),
        "fallback_used": summarizer is None,
    }


def _analyze_sentiment(text: str) -> Dict[str, float | str | bool]:
    sentiment_pipeline = _get_sentiment()
    if sentiment_pipeline is not None:
        try:
            output = sentiment_pipeline(text[:2000])[0]
            raw_label = str(output.get("label", "neutral")).upper()
            score = float(output.get("score", 0.0))
            if raw_label in {"POSITIVE", "LABEL_2"}:
                label = "positive"
            elif raw_label in {"NEGATIVE", "LABEL_0"}:
                label = "negative"
            else:
                label = "neutral"
            return {"label": label, "score": round(score, 4), "fallback_used": False}
        except Exception:
            pass
    lex = _lexicon_sentiment(text)
    return {"label": lex["label"], "score": lex["score"], "fallback_used": True}


@router.post("/text")
def summarize_text(payload: SummarizeIn):
    text = _normalize_whitespace(payload.text)
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    result = _summarize_large_text(
        text=text,
        min_length=payload.min_length,
        max_length=payload.max_length,
    )

    return {
        "type": "summary",
        "language": payload.language.strip().lower(),
        "model": summary_model_name,
        "chunks": result["chunks"],
        "fallback_used": result["fallback_used"],
        "summary": result["summary"],
    }


@router.post("/news")
def summarize_news(payload: NewsSummarizeIn):
    body = _normalize_whitespace(payload.text)
    if not body:
        raise HTTPException(status_code=400, detail="text is empty")

    combined = _normalize_whitespace(f"{payload.title}. {body}" if payload.title.strip() else body)
    result = _summarize_large_text(
        text=combined,
        min_length=payload.min_length,
        max_length=payload.max_length,
    )

    sentiment_text = _normalize_whitespace(f"{payload.title} {result['summary']}")
    sentiment = _analyze_sentiment(sentiment_text)

    mode = payload.impact_mode.strip().lower()
    allowed_modes = {"all", "industry", "sector", "stock"}
    if mode not in allowed_modes:
        raise HTTPException(status_code=400, detail="impact_mode must be one of: all, industry, sector, stock")

    impact = {
        "mode": mode,
        "industry": None,
        "sector": None,
        "stocks": [],
    }

    if mode in {"all", "industry"}:
        industry_text = _filter_text_by_keywords(combined, payload.industry_keywords)
        if industry_text:
            impact["industry"] = {
                "keywords": payload.industry_keywords,
                "sentiment": _analyze_sentiment(industry_text),
            }

    if mode in {"all", "sector"}:
        sector_text = _filter_text_by_keywords(combined, payload.sector_keywords)
        if sector_text:
            impact["sector"] = {
                "keywords": payload.sector_keywords,
                "sentiment": _analyze_sentiment(sector_text),
            }

    if mode in {"all", "stock"}:
        tickers = [s.strip().upper() for s in payload.stocks if s.strip()]
        if not tickers:
            tickers = _extract_tickers(combined)
        for ticker in tickers:
            stock_text = _filter_text_by_keywords(combined, [ticker])
            target = stock_text if stock_text else sentiment_text
            impact["stocks"].append(
                {
                    "ticker": ticker,
                    "sentiment": _analyze_sentiment(target),
                }
            )

    return {
        "type": "news_summary",
        "language": payload.language.strip().lower(),
        "summary_model": summary_model_name,
        "sentiment_model": sentiment_model_name,
        "chunks": result["chunks"],
        "summary_fallback_used": result["fallback_used"],
        "summary": result["summary"],
        "sentiment": sentiment,
        "impact": impact,
    }
