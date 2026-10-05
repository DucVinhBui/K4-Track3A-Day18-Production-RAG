from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    zeros = {m: 0.0 for m in metric_names}
    # RAGAS cần OPENAI_API_KEY (LLM judge) và Python 3.11+
    try:
        import math
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        def _num(x) -> float:
            try:
                x = float(x)
            except (TypeError, ValueError):
                return 0.0
            return 0.0 if math.isnan(x) else x

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                **{m: _num(row.get(m, 0.0)) for m in metric_names},
            )
            for _, row in df.iterrows()
        ]
        # Trung bình bỏ qua NaN (RAGAS trả NaN khi judge parse lỗi)
        aggregate = {}
        for m in metric_names:
            vals = [float(v) for v in df[m] if v is not None and not math.isnan(float(v))]
            aggregate[m] = round(sum(vals) / len(vals), 4) if vals else 0.0
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {**zeros, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — answer không bám context",
                         "Tighten prompt, lower temperature, bắt buộc trích dẫn context"),
        "context_recall": ("Missing relevant chunks — retriever bỏ sót thông tin",
                           "Improve chunking (parent-child), thêm BM25/query expansion, tăng top_k"),
        "context_precision": ("Too many irrelevant chunks — context nhiễu",
                              "Add reranking, metadata filter (version/category), giảm top_k"),
        "answer_relevancy": ("Answer doesn't match question — trả lời lan man/lạc đề",
                             "Improve prompt template: trả lời trực tiếp, ngắn gọn"),
    }
    metric_names = list(diagnostic_tree.keys())

    analyzed = []
    for r in eval_results:
        scores = {m: float(getattr(r, m)) for m in metric_names}
        avg = sum(scores.values()) / len(scores)
        worst_metric = min(scores, key=scores.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "scores": {m: round(v, 4) for m, v in scores.items()},
            "avg_score": round(avg, 4),
            "worst_metric": worst_metric,
            "score": round(scores[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
