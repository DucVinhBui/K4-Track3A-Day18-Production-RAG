# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Bùi Đức Vinh
**MSSV:** 2A202602801
**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 05/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 với all-MiniLM-L6-v2 tạo **208 chunks** (avg 99 ký tự, min 6) vs basic **51 chunks** (avg 410). Model tiếng Anh cho cosine thấp giữa các câu tiếng Việt liên tiếp → cắt quá vụn; cần model đa ngữ (bge-m3) hoặc hạ threshold. Vì vậy pipeline chọn hierarchical (99 children avg 210, max 256) + structure-aware (106 chunks theo header) ổn định hơn. |
| Hierarchical (parent-child) | M1 + pipeline | `chunk_hierarchical()`, `run_query()` | Retrieve child 256 ký tự (precision) → trả **parent** cho LLM (context đầy đủ). Phải prefix `parent_id` bằng tên file vì mỗi tài liệu đều sinh `parent_0` → trùng ID khi gộp corpus. Context precision 0.9667. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | underthesea nối từ ghép bằng `_` ("nghỉ_phép") → phải `replace("_", " ")` + lowercase + bỏ token dấu câu thì query "nghỉ phép" mới khớp. RRF (k=60) chỉ dùng thứ hạng nên không cần chuẩn hóa thang điểm BM25 (0-20) với cosine (0-1). Hybrid search chỉ tốn **89 ms/query**. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 rerank 20 candidates: **p50 572 ms**, avg 878 ms (query đầu 6.7s do warm-up) — chiếm ~40% latency. Top-1 đúng tài liệu cho **20/20** câu hỏi. Cache model theo tên giúp test suite không load lại ~2GB weights mỗi lần (37 tests trong ~32s). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Production: F 0.896 · AR 0.857 · CP 0.967 · CR 0.933 (baseline 0.844 / 0.678 / 0.925 / 0.925). Answer relevancy thấp nhất ở baseline vì câu trả lời dài dòng → sửa prompt "trả lời trực tiếp trước" tăng +0.18. Phát hiện judge false-negative: answer trích nguyên văn vẫn có faithfulness 0.0 (câu malware). |
| Contextual embeddings | M5 | `_enrich_single_call()` (combined), `contextual_prepend()` | 1 call/chunk trả về summary + questions + context + metadata (JSON mode). Gửi kèm **toàn văn parent** để LLM viết câu context chính xác (tên chính sách, phiên bản, hiện hành/đã thay thế) — đúng tinh thần Anthropic contextual retrieval. 104 chunks enrich trong 43s nhờ 8 luồng; cache trên đĩa theo hash → chạy lại không tốn API. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

**1. Docker/Qdrant không chạy**
- **Exact error:** `Cannot connect to the Docker daemon at unix:///Users/ducvinhbui/.docker/run/docker.sock. Is the docker daemon running?`, và khi chạy pipeline: `UserWarning: Failed to obtain server version. Unable to check client-server compatibility.`
- **Debug:** `DenseSearch.__init__` thử kết nối Qdrant với `timeout=2`, nếu lỗi thì fallback `QdrantClient(":memory:")`. Corpus chỉ 104 chunks nên in-memory đủ dùng.
- **Bài học:** production cần health-check rõ ràng thay vì fallback im lặng — nếu không sẽ không biết đang dùng index tạm.

**2. API Qdrant thay đổi giữa các version**
- **Vấn đề:** scaffold gợi ý `recreate_collection()` (deprecated trong qdrant-client 1.19) và `search()` (đã thay bằng `query_points()`).
- **Giải quyết:** dùng `collection_exists()` → `delete_collection()` → `create_collection()`, search bằng `query_points(...).points`. Upsert theo batch 256 điểm.

**3. Trùng `parent_id` giữa các tài liệu**
- **Triệu chứng:** pipeline chunk từng tài liệu riêng → mọi tài liệu đều có `parent_0`, `parent_1` → map child → parent sẽ trả nhầm tài liệu.
- **Giải quyết:** `pid = f"{source}::parent_{n}"` và lưu luôn `parent_text` vào metadata của child (payload Qdrant) để `run_query()` lấy parent mà không cần lookup thêm. Test `test_hierarchical_valid_parent_ids` vẫn pass.

**4. PDF scan không có text layer**
- **Exact output:** `Ignoring wrong pointing object 11 0 (offset 0)` và `⚠️ Bỏ qua BCTC.pdf: PDF scan ảnh, không có text layer (cần OCR).`
- **Phân tích:** 2/3 PDF (`BCTC.pdf`, Nghị định 13/2023) là ảnh scan → pypdf trả chuỗi rỗng. Test set không hỏi về 2 file này nên không ảnh hưởng điểm, nhưng thực tế sẽ là "silent data loss".
- **Hướng bổ sung:** OCR (Tesseract `vie` / PaddleOCR) hoặc vision LLM trong bước ingest.

**5. Placeholder API key**
- **Vấn đề:** `cp .env.example .env` tạo `OPENAI_API_KEY=sk-...` → code kiểm tra `if OPENAI_API_KEY:` coi là có key → mọi call sẽ lỗi 401 và retry (RAGAS `max_retries=10`) rất lâu.
- **Giải quyết:** kiểm tra độ dài key trước khi chạy; mọi hàm M5 đều có fallback (extractive summary/questions, prefix theo tên tài liệu) để test vẫn pass khi không có key.

**6. RAGAS có thể trả NaN (xử lý phòng ngừa)**
- **Vấn đề:** khi LLM judge parse lỗi, RAGAS trả `NaN` → `json.dump` ghi `NaN` (JSON không hợp lệ) và làm hỏng trung bình.
- **Giải quyết:** aggregate bằng trung bình bỏ qua NaN, per-question thay NaN bằng 0.0.

**Kiến thức còn thiếu & cách bổ sung:**
- Cách RAGAS tính context_precision (average precision có trọng số theo rank) → đọc source `ragas/metrics/_context_precision.py` để hiểu vì sao đặt bản superseded ở rank 1 làm điểm còn 0.5.
- Độ tin cậy của LLM-as-judge với tiếng Việt → cần đọc thêm về `ragas.adapt()` và so sánh với chấm tay trên mẫu nhỏ.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý RAG tra cứu tài liệu nghiên cứu cho luận văn

#### 1. Hiện trạng
- **Pipeline hiện tại:** thư viện ~100 paper PDF (tiếng Anh) + ghi chú thí nghiệm (markdown, tiếng Việt/Anh). Hiện đang tìm bằng keyword search trong Zotero và đọc thủ công; prototype RAG đơn giản (chia đoạn cố định 1000 ký tự, dense search, top-3) trả lời kém.
- **Vấn đề / Bottlenecks:**
  - Chunk cố định cắt ngang bảng kết quả và công thức → mất số liệu.
  - Câu hỏi so sánh nhiều paper ("phương pháp A vs B trên dataset X") thiếu tài liệu thứ hai (multi-hop, giống lỗi #4 trong lab).
  - Nhiều version của cùng một paper (arXiv v1/v2, bản hội nghị) → trích số liệu cũ (giống lỗi version #3, #5).
  - Thuật ngữ/viết tắt chuyên ngành (tên dataset, tên metric) dense search hay bỏ sót.
  - Không có cách đo chất lượng câu trả lời.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Structure-aware theo section của paper (Abstract, Method, Experiments…) làm parent + child 256-512 token để retrieve; giữ nguyên bảng/caption trong một chunk. Lý do: lab cho thấy retrieve child + trả parent cho context precision 0.97.
2. **Search retrieval:** Hybrid BM25 + dense (bge-m3) + RRF — BM25 bắt chính xác tên dataset, viết tắt, con số; dense bắt paraphrase. Thêm query decomposition cho câu hỏi so sánh nhiều paper và MMR theo `source` để đa dạng tài liệu.
3. **Reranking:** Có — bge-reranker-v2-m3 trên top-20, ngưỡng score để loại context nhiễu. Corpus nhỏ, dùng cá nhân nên latency ~0.6s chấp nhận được.
4. **Evaluation:** Tự soạn golden set ~40 câu (lookup số liệu, so sánh, định nghĩa, câu hỏi "paper nào…") → chạy RAGAS 4 metrics mỗi lần đổi chunking/prompt; kiểm tra tay các câu điểm 0 để loại false-negative của judge.
5. **Enrichment:** Combined single-call: contextual prepend (tên paper, năm, section) + HyQA + metadata (`title`, `year`, `venue`, `version`, `datasets`) → filter/boost theo năm và bản mới nhất.

#### 3. Timeline triển khai
- **Tuần 1:** Ingest PDF (pypdf + OCR cho bản scan), parse section; soạn golden set 40 câu; đo baseline RAGAS.
- **Tuần 2:** Structure-aware + hierarchical chunking; hybrid BM25 + dense + RRF; đo lại.
- **Tuần 3:** Reranker + combined enrichment với metadata paper; boost bản mới nhất.
- **Tuần 4:** Query decomposition cho câu hỏi so sánh, trích dẫn nguồn (paper + trang) trong câu trả lời, viết báo cáo trước/sau.
