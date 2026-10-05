from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY

LLM_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """Bạn là trợ lý chính sách nội bộ công ty. Trả lời CHỈ dựa trên context được cung cấp.
Quy tắc:
- Bắt đầu bằng câu trả lời trực tiếp (Có/Không/con số/người phê duyệt), sau đó giải thích ngắn gọn.
- Nếu context có nhiều phiên bản chính sách, dùng phiên bản HIỆN HÀNH (mới nhất) và nêu rõ phiên bản cũ đã bị thay thế.
- Với câu hỏi cần tính toán, trình bày phép tính dựa trên số liệu trong context.
- Không thêm thông tin ngoài context. Nếu context không có thông tin → trả lời 'Không tìm thấy.'"""


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_text = {p.metadata["parent_id"]: p.text for p in parents}
        for child in children:
            # Lưu parent_text vào metadata: retrieve child (precision) → trả parent (context)
            all_chunks.append({"text": child.text, "metadata": {
                **child.metadata, "parent_id": child.parent_id,
                "parent_text": parent_text[child.parent_id]}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def _generate(query: str, contexts: list[str]) -> str:
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n---\n\n".join(contexts)
            resp = client.chat.completions.create(model=LLM_MODEL, temperature=0, messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            return contexts[0]
    return contexts[0] if contexts else "Không tìm thấy thông tin."


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              timings: dict | None = None) -> tuple[str, list[str]]:
    """Run single query through pipeline: hybrid → rerank children → parent contexts → LLM."""
    timings = timings if timings is not None else {}

    t0 = time.perf_counter()
    results = search.search(query)
    timings["search_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=len(docs))
    timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

    # Child → parent, bỏ trùng parent, giữ thứ tự theo rerank score
    contexts, seen = [], set()
    for r in reranked or results:
        parent = r.metadata.get("parent_text") or r.text
        if parent not in seen:
            seen.add(parent)
            contexts.append(parent)
        if len(contexts) >= RERANK_TOP_K:
            break

    t0 = time.perf_counter()
    answer = _generate(query, contexts)
    timings["llm_ms"] = (time.perf_counter() - t0) * 1000
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    latencies = []
    for i, item in enumerate(test_set):
        timings: dict = {}
        answer, contexts = run_query(item["question"], search, reranker, timings)
        latencies.append(timings)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    _report_latency(latencies)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    _save_answers(results.get("per_question", []), questions, answers, all_contexts, ground_truths)
    return results


def _report_latency(latencies: list[dict]) -> None:
    """In bảng latency breakdown từng bước + lưu reports/latency_report.json."""
    steps = ["search_ms", "rerank_ms", "llm_ms"]
    summary = {}
    for step in steps:
        vals = sorted(t[step] for t in latencies if step in t)
        if vals:
            summary[step] = {"avg": round(sum(vals) / len(vals), 1),
                             "p50": round(vals[len(vals) // 2], 1),
                             "max": round(vals[-1], 1)}
    totals = sorted(sum(t.get(s, 0) for s in steps) for t in latencies)
    if totals:
        summary["total_ms"] = {"avg": round(sum(totals) / len(totals), 1),
                               "p50": round(totals[len(totals) // 2], 1), "max": round(totals[-1], 1)}

    print(f"\n[Latency] {'Step':<12} {'avg (ms)':>10} {'p50 (ms)':>10} {'max (ms)':>10}")
    for step, s in summary.items():
        print(f"          {step.replace('_ms', ''):<12} {s['avg']:>10.1f} {s['p50']:>10.1f} {s['max']:>10.1f}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/latency_report.json", "w", encoding="utf-8") as f:
        json.dump({"num_queries": len(latencies), "breakdown": summary}, f, ensure_ascii=False, indent=2)


def _save_answers(per_question, questions, answers, contexts, ground_truths) -> None:
    """Lưu chi tiết từng câu (answer, contexts, scores) để viết failure analysis."""
    scores = {r.question: r for r in per_question}
    rows = []
    for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
        r = scores.get(q)
        rows.append({
            "question": q, "answer": a, "ground_truth": gt, "contexts": c,
            "scores": {m: round(getattr(r, m), 4) for m in
                       ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]} if r else {},
        })
    os.makedirs("reports", exist_ok=True)
    with open("reports/per_question.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
