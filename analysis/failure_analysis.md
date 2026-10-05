# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Bùi Đức Vinh
**MSSV:** 2A202602801
**Khóa:** K4 - Track 3A

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8444 | **0.8960** | +0.0516 |
| Answer Relevancy | 0.6782 | **0.8569** | +0.1787 |
| Context Precision | 0.9250 | **0.9667** | +0.0417 |
| Context Recall | 0.9250 | **0.9333** | +0.0083 |

- **Naive baseline:** paragraph chunking (500 ký tự) → dense-only (bge-m3) top-3 → gpt-4o-mini, prompt mặc định.
- **Production:** hierarchical chunking (child 256 / parent 2048) → M5 combined enrichment (1 call/chunk, có toàn văn tài liệu làm ngữ cảnh) → hybrid BM25 (underthesea) + dense + RRF (top-20) → rerank bge-reranker-v2-m3 → map child → **parent** (dedupe, top-3) → gpt-4o-mini (temperature 0, prompt ưu tiên phiên bản hiện hành, trả lời trực tiếp).
- Cả 4 metrics ≥ 0.75; Faithfulness ≥ 0.85.
- Cải thiện lớn nhất là **Answer Relevancy (+0.18)**: prompt mới buộc trả lời thẳng (Có/Không/con số) ở câu đầu, thay vì diễn giải dài.
- Context Recall gần như không đổi: corpus nhỏ (26 tài liệu) nên dense-only đã tìm đúng tài liệu; phần còn thiếu là các câu cần **tài liệu thứ 2** (multi-hop / phiên bản cũ) — xem #4, #5.

### Latency breakdown (20 queries, Apple M3 Pro, CPU/MPS, Qdrant in-memory)

| Bước | avg (ms) | p50 (ms) | max (ms) |
|------|---------:|---------:|---------:|
| Hybrid search (BM25 + dense + RRF) | 89.0 | 86.0 | 205.3 |
| Rerank (cross-encoder, 20 candidates) | 877.6 | 571.7 | 6711.6 |
| LLM generation (gpt-4o-mini) | 1260.1 | 1181.2 | 2388.1 |
| **Tổng / query** | **2226.6** | **1887.8** | **8479.4** |

Offline: enrichment 104 chunks = 43.0s (8 luồng song song), indexing = 10.8s. Max rerank 6.7s là query đầu tiên (warm-up model); p50 mới phản ánh latency thực.

---

## Bottom-5 Failures

> Xếp theo trung bình 4 metrics (thấp → cao), nguồn: `reports/ragas_report.json` + `reports/per_question.json`.

### #1 — Tính phí tạm ứng quá hạn (numeric)
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày → quá hạn 5 ngày; 2%/tháng × 15tr = 300.000đ/tháng → pro-rata ≈ 50.000đ cho 5 ngày.
- **Got:** "400.000 VNĐ" — LLM tính 300.000đ cho "tháng đầu tiên" + 100.000đ cho 5 ngày tháng thứ hai.
- **Scores:** F 0.14 · AR 0.90 · CP 1.00 · CR 0.67
- **Worst metric:** faithfulness (0.14)
- **Error Tree:** Output sai → Context đúng? **Có** (`tam_ung.md` rank 1, chứa "15 ngày" và "2%/tháng") → Query OK? **Có** → **Lỗi ở bước Generation (reasoning số học)**: LLM tính phí từ ngày nhận tiền thay vì từ ngày quá hạn.
- **Root cause:** gpt-4o-mini làm phép tính nhiều bước không chặt chẽ; prompt chỉ nói "trình bày phép tính" nhưng không ép xác định mốc thời gian.
- **Suggested fix:** (1) Prompt chain-of-thought có cấu trúc: "Bước 1: xác định thời hạn; Bước 2: số ngày quá hạn; Bước 3: áp công thức"; (2) dùng tool/calculator (function calling) cho phép tính; (3) dùng model reasoning mạnh hơn cho query loại numeric (route theo query type).

### #2 — Malware có nên tự xử lý? (negation)
- **Question:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** KHÔNG. Phải báo cáo trong 1 giờ qua helpdesk@cty.vn/hotline; tự xử lý là vi phạm nghiêm trọng.
- **Got:** "Không. Nhân viên tuyệt đối **không tự ý xử lý malware** hoặc tìm cách khắc phục mà không có sự hướng dẫn của đội CNTT."
- **Scores:** F 0.00 · AR 0.85 · CP 1.00 · CR 1.00
- **Worst metric:** faithfulness (0.00)
- **Error Tree:** Output sai? **Không** — câu trả lời gần như trích nguyên văn `bao_mat_su_co.md` → Context đúng? **Có** (rank 1) → **Lỗi ở bước Evaluation (false negative của RAGAS judge)**.
- **Root cause:** Faithfulness = 0 trong khi answer được trích nguyên văn → nhiều khả năng RAGAS (LLM judge, prompt tiếng Anh) tách statement / NLI thất bại với câu tiếng Việt có markdown `**bold**` và phủ định "Không". Đồng thời answer thiếu ý "báo cáo trong 1 giờ" → không sai nhưng chưa đầy đủ.
- **Suggested fix:** (1) Strip markdown khỏi answer trước khi chấm; (2) dùng judge mạnh hơn (gpt-4o) hoặc adapt prompt RAGAS sang tiếng Việt (`ragas.adapt`); (3) prompt generation yêu cầu nêu cả **hành động cần làm** với câu hỏi "có nên…không"; (4) spot-check thủ công các điểm 0.0.

### #3 — Thâm niên cộng phép (version conflict)
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** v2024 hiện hành: từ 3 năm, +1 ngày mỗi 3 năm (v2023 cũ yêu cầu 5 năm).
- **Got:** "3 năm… theo chính sách nghỉ phép năm phiên bản 2024." (đúng, nhưng không nhắc phiên bản cũ)
- **Scores:** F 0.75 · AR 0.78 · CP 0.50 · CR 1.00
- **Worst metric:** context_precision (0.50)
- **Error Tree:** Output đúng → Context đúng? **Có nhưng sai thứ tự**: `nghi_phep_nam_v2023.md` (superseded) đứng **rank 1**, v2024 rank 2 (rerank score đều ≈ 0.99) → Query OK? Có → **Lỗi ở bước Ranking (thiếu tín hiệu version/recency)**.
- **Root cause:** Cross-encoder chỉ đo độ liên quan ngữ nghĩa; hai phiên bản gần như giống nhau về nội dung nên thứ tự là ngẫu nhiên. Pipeline không có metadata `version`/`effective_date`/`superseded` để ưu tiên bản hiện hành.
- **Suggested fix:** (1) Trích metadata `effective_date`, `status: current|superseded` lúc ingest (M5 auto-metadata); (2) boost điểm bản hiện hành sau rerank (hoặc tie-break theo ngày hiệu lực); (3) đặt bản superseded sau cùng và gắn nhãn "[ĐÃ BỊ THAY THẾ]" trong context.

### #4 — Senior 9 năm: phép + lương (multi-hop)
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép; lương Senior (P3-P4) 20-35 triệu/tháng.
- **Got:** "18 ngày phép năm… không có thông tin cụ thể về mức lương trong context."
- **Scores:** F 0.71 · AR 0.96 · CP 1.00 · CR 0.50
- **Worst metric:** context_recall (0.50)
- **Error Tree:** Output thiếu ý → Context đúng? **Thiếu**: top-3 = nghỉ phép v2024, v2023, nghỉ không lương; **không có `bang_luong_2024.md`** → Query OK? **Không** — 1 query chứa 2 intent (phép + lương), embedding/rerank bị intent "nghỉ phép" chi phối → **Lỗi ở bước Query understanding / Retrieval**.
- **Root cause:** Multi-hop question; 3 slot context bị chiếm bởi các tài liệu cùng chủ đề nghỉ phép (2 phiên bản + nghỉ không lương). LLM đã làm đúng khi nói "không có thông tin" (không hallucinate).
- **Suggested fix:** (1) Query decomposition: LLM tách thành sub-queries ("số ngày phép thâm niên 9 năm", "khung lương Senior") → retrieve riêng → merge; (2) diversity (MMR) theo `source` khi chọn top-3; (3) tăng top-k context lên 4-5 cho query multi-hop.

### #5 — MFA có bắt buộc không? (version / recall)
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo v2.0 hiện hành, bắt buộc cho email, VPN, hệ thống nội bộ. v1.0 cũ không yêu cầu MFA.
- **Got:** "Có. Tất cả nhân viên bắt buộc phải kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ."
- **Scores:** F 1.00 · AR 0.89 · CP 1.00 · CR 0.50
- **Worst metric:** context_recall (0.50)
- **Error Tree:** Output đúng (với chính sách hiện hành) → Context đúng? **Thiếu một phần**: có `mat_khau_v2.md` nhưng không có `mat_khau_v1.md` (rank 2-3 là mua sắm, WFH — rerank score ≈ 0.00) → Query OK? Có, nhưng query không nhắc "phiên bản cũ" → **Lỗi ở bước Retrieval (không link tài liệu superseded)**.
- **Root cause:** v1.0 không chứa từ "MFA" nên không match cả BM25 lẫn dense; ground truth lại yêu cầu so sánh với bản cũ. Hai slot context còn lại bị lấp bằng tài liệu không liên quan (score ≈ 0).
- **Suggested fix:** (1) Metadata `supersedes: mat_khau_v1.md` → khi retrieve bản mới thì kéo kèm bản cũ (document linking); (2) ngưỡng rerank score (vd < 0.05 thì bỏ) để không đưa context nhiễu; (3) hoặc chấp nhận — đây là trường hợp ground truth đòi hỏi nhiều hơn câu hỏi.

---

## Tổng hợp theo Error Tree

| Bước lỗi | Câu | Fix ưu tiên |
|----------|-----|-------------|
| Generation (reasoning) | #1 | CoT có cấu trúc / calculator tool |
| Evaluation (judge) | #2 | Strip markdown, judge tiếng Việt / model mạnh hơn |
| Ranking (version) | #3 | Metadata version + boost bản hiện hành |
| Query / Retrieval (multi-hop) | #4 | Query decomposition + MMR theo source |
| Retrieval (document linking) | #5 | `supersedes` link + rerank threshold |

→ 3/5 lỗi liên quan tới **quản lý phiên bản tài liệu và multi-hop** chứ không phải chunking/embedding. Đây là hướng cải tiến có ROI cao nhất.

## Case Study (cho presentation)

**Question chọn phân tích:** #4 — "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"

**Error Tree walkthrough:**
1. **Output đúng?** → Một nửa: phần phép năm đúng (18 ngày, tính đúng 15 + 9÷3), phần lương thiếu. LLM trung thực ("không có thông tin") nên faithfulness vẫn ổn → lỗi không nằm ở generation.
2. **Context đúng?** → Không đủ: 3 context đều là tài liệu nghỉ phép; `bang_luong_2024.md` không lọt top-3 (context_recall 0.50).
3. **Query rewrite OK?** → Không: pipeline dùng nguyên câu hỏi 2 intent làm 1 query; vector query bị kéo về phía "nghỉ phép" vì từ khóa này chiếm ưu thế, và reranker chấm cả câu hỏi với từng chunk nên chunk lương chỉ khớp một nửa.
4. **Fix ở bước:** **Query transformation** — thêm bước decomposition trước M2: tách sub-queries → hybrid search + rerank cho từng sub-query → lấy top-2 mỗi sub-query → merge, dedupe theo parent. Kỳ vọng context_recall của câu này từ 0.50 → 1.0 mà không ảnh hưởng câu single-hop (classifier chỉ decompose khi phát hiện nhiều intent).

**Nếu có thêm 1 giờ, sẽ optimize:**
- Query decomposition cho câu multi-hop (#4) + MMR đa dạng nguồn.
- Metadata `version` / `effective_date` / `supersedes` trong M5 → boost bản hiện hành và kéo kèm bản cũ khi cần so sánh (#3, #5).
- Ngưỡng rerank score để loại context nhiễu (score ≈ 0) → giảm token và nguy cơ LLM bị phân tâm.
- Chạy RAGAS 3 lần lấy trung bình, strip markdown trước khi chấm để giảm nhiễu judge (#2).
