from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

LLM_MODEL = "gpt-4o-mini"
CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          ".cache", "enrichment.json")


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _chat(system: str, user: str, max_tokens: int, json_mode: bool = False) -> str:
    """Gọi OpenAI chat completion (temperature=0 để kết quả ổn định)."""
    from openai import OpenAI
    client = OpenAI()
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0,
        **kwargs,
    )
    return resp.choices[0].message.content.strip()


def _extractive_summary(text: str) -> str:
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]).rstrip(".") + "." if sentences else text


def _extractive_questions(text: str, n_questions: int) -> list[str]:
    import re
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text)
                 if len(s.strip()) > 10 and not s.strip().startswith(("#", ">"))]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENAI_API_KEY:
        try:
            return _chat("Tóm tắt đoạn văn sau trong 1-2 câu ngắn gọn bằng tiếng Việt, "
                         "giữ nguyên các con số quan trọng, không dài hơn đoạn gốc.", text, max_tokens=150)
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")
    # Extractive fallback (không cần API)
    return _extractive_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if OPENAI_API_KEY:
        try:
            content = _chat(f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể "
                            "trả lời. Trả về mỗi câu hỏi trên 1 dòng, không đánh số.", text, max_tokens=200)
            questions = [q.strip().lstrip("0123456789.-) ") for q in content.split("\n") if q.strip()]
            return questions[:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")
    # Extractive fallback
    return _extractive_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if OPENAI_API_KEY:
        try:
            context = _chat("Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
                            "Chỉ trả về 1 câu.",
                            f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", max_tokens=80)
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")
    # Simple fallback
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    default = {"topic": "general", "entities": [], "category": "policy", "language": "vi"}
    if OPENAI_API_KEY:
        try:
            import json as _json
            content = _chat('Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], '
                            '"category": "policy|hr|it|finance", "language": "vi|en"}',
                            text, max_tokens=150, json_mode=True)
            return {**default, **_json.loads(content)}
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")
    return default


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str, document: str = "") -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    `document` (tùy chọn): toàn văn parent/tài liệu chứa chunk — giúp LLM viết context
    chính xác hơn (Anthropic contextual retrieval).
    """
    if OPENAI_API_KEY:
        try:
            import json as _json
            system = """Bạn hỗ trợ indexing cho hệ thống RAG tiếng Việt. Phân tích ĐOẠN VĂN (thuộc TÀI LIỆU) và trả về JSON:
{
  "summary": "tóm tắt 1-2 câu, giữ nguyên các con số",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1-2 câu cho biết đoạn văn thuộc tài liệu/chính sách nào (kèm phiên bản, hiện hành hay đã bị thay thế nếu có) và nói về chủ đề gì",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""
            doc_block = f"\n\nTÀI LIỆU:\n{document[:3000]}" if document and document != text else ""
            user = f"Tên file: {source}{doc_block}\n\nĐOẠN VĂN:\n{text}"
            return _json.loads(_chat(system, user, max_tokens=400, json_mode=True))
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")
    return {}


def _load_cache() -> dict:
    import json as _json
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return _json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    import json as _json
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        _json.dump(cache, f, ensure_ascii=False)


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    combined_results: dict[int, dict] = {}
    if use_combined:
        # 1 call/chunk, chạy song song + cache trên đĩa (chạy lại không tốn API)
        import hashlib
        from concurrent.futures import ThreadPoolExecutor

        cache = _load_cache() if OPENAI_API_KEY else {}

        def _key(chunk: dict) -> str:
            meta = chunk.get("metadata", {})
            raw = f"{LLM_MODEL}|{meta.get('source', '')}|{meta.get('parent_text', '')}|{chunk['text']}"
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()

        def _work(i: int) -> dict:
            chunk = chunks[i]
            meta = chunk.get("metadata", {})
            key = _key(chunk)
            if key in cache:
                return cache[key]
            result = _enrich_single_call(chunk["text"], meta.get("source", ""), meta.get("parent_text", ""))
            if result:
                cache[key] = result
            return result

        with ThreadPoolExecutor(max_workers=8 if OPENAI_API_KEY else 1) as pool:
            for i, result in enumerate(pool.map(_work, range(len(chunks)))):
                combined_results[i] = result
                if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                    print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)
        if OPENAI_API_KEY:
            _save_cache(cache)

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = combined_results[i]
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if not use_combined and ((i + 1) % 10 == 0 or (i + 1) == len(chunks)):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
