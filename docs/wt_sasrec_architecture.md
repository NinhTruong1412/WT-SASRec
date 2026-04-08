# WTSASRec — Architecture & Implementation Analysis

> **Thesis contribution:** *"Introduce watch time as a weighting or gating mechanism into a baseline model — items watched longer get higher influence on the next prediction."*  
> **Approach used:** Sửa đổi attention score (Attention Score Modification)

---

## 1. Tổng quan cơ chế (Overview)

WTSASRec mở rộng SASRec bằng cách **thêm một bias phụ thuộc vào watch time vào attention score trước khi tính softmax**. Điều này khiến mỗi query tự nhiên chú ý nhiều hơn đến các item mà người dùng đã xem lâu hơn trong lịch sử.

```
Standard SASRec:
  scores = QKᵀ / √d + causal_mask
  output = softmax(scores) · V

WTSASRec:
  scores = QKᵀ / √d + causal_mask + α · wt_bias
  output = softmax(scores) · V
```

`α` là một **learnable scalar parameter** (khởi tạo = 1.0), cho phép model tự học mức độ ảnh hưởng của watch time.

---

## 2. Luồng dữ liệu đầy đủ (Full Data Flow)

```
Input batch:
  item_seq       : (B, L)       — item ID sequence
  watch_time_list: (B, L)       — raw watch time (seconds) per position

────────────────────────────────────────────────────────────────────
STEP 1 — Watch-Time Normalization  [WTSASRec._normalize_watch_time]
────────────────────────────────────────────────────────────────────

  log_wt  = log(1 + watch_time)            (B, L)   — log scale
  max_wt  = max(log_wt, dim=1)             (B, 1)   — per-sequence max
  wt_norm = log_wt / (max_wt + 1e-8)      (B, L)   — in [0, 1]

  Rationale: log scale prevents outliers; per-sequence max keeps [0,1].

────────────────────────────────────────────────────────────────────
STEP 2 — Watch-Time Bias Projection  [WTSASRec._compute_wt_bias]
────────────────────────────────────────────────────────────────────

  wt_norm.unsqueeze(-1)                    (B, L, 1)
  wt_proj = Linear(1 → H_heads)
  wt_bias = wt_proj(wt_norm.unsqueeze(-1)) (B, L, H_heads)

  Each attention head gets its own learnable watch-time bias coefficient.

────────────────────────────────────────────────────────────────────
STEP 3 — Standard SASRec Embedding  [WTSASRec.forward]
────────────────────────────────────────────────────────────────────

  item_emb   = ItemEmbedding(item_seq)     (B, L, H)
  pos_emb    = PositionEmbedding(0..L-1)  (B, L, H)
  input_emb  = LayerNorm(Dropout(
                  item_emb + pos_emb))    (B, L, H)

────────────────────────────────────────────────────────────────────
STEP 4 — Watch-Time Augmented Attention  [WTMultiHeadAttention]
────────────────────────────────────────────────────────────────────

  Q = Linear(H → H)(input_emb)            (B, L, H)
  K = Linear(H → H)(input_emb)            (B, L, H)
  V = Linear(H → H)(input_emb)            (B, L, H)

  Reshape to multi-head:
  Q → (B, H_heads, L, d_head)   where d_head = H / H_heads
  K → (B, H_heads, d_head, L)   [transposed for matmul]
  V → (B, H_heads, L, d_head)

  Raw attention scores:
  A = Q · K / √d_head            (B, H_heads, L_q, L_k)

  Add causal mask (future positions → -inf):
  A = A + causal_mask             (B, 1, 1, L) broadcasts

  ★ Add watch-time bias:
  wt_bias_t = wt_bias
              .permute(0,2,1)     (B, H_heads, L_k)
              .unsqueeze(2)       (B, H_heads, 1, L_k)
  A = A + α · wt_bias_t          (B, H_heads, L_q, L_k)
                                  ↑ broadcasts over all query positions

  Normalize:
  probs   = softmax(A)            (B, H_heads, L_q, L_k)
  context = probs · V             (B, H_heads, L_q, d_head)

  Merge heads + project:
  context → (B, L, H)
  output  = LayerNorm(Dropout(dense(context)) + input_emb)

────────────────────────────────────────────────────────────────────
STEP 5 — Feed-Forward + Stacking  [WTTransformerEncoder]
────────────────────────────────────────────────────────────────────

  output = FeedForward(attention_output)   (B, L, H)
  [repeat for n_layers transformer blocks — same wt_bias passed to all]

────────────────────────────────────────────────────────────────────
STEP 6 — Gather last position + BPR Loss
────────────────────────────────────────────────────────────────────

  seq_repr = output[:, seq_len-1, :]      (B, H)   — last real position
  score    = seq_repr · item_emb          (B,)
  loss     = BPR(pos_score, neg_score)
```

---

## 3. Sơ đồ kiến trúc (Architecture Diagram)

```
                     item_seq (B,L)         watch_time_list (B,L)
                         │                          │
               ┌─────────┴──────────┐    ┌──────────┴──────────┐
               │  ItemEmbedding(L,H)│    │  log(1+wt)/max_wt   │  STEP 1
               │  + PosEmbedding    │    │  → wt_norm (B,L)    │
               │  + LayerNorm       │    └──────────┬──────────┘
               └─────────┬──────────┘               │
                         │  (B,L,H)      ┌──────────┴──────────┐
                         │               │  Linear(1→H_heads)  │  STEP 2
                         │               │  → wt_bias (B,L,Hh) │
                         │               └──────────┬──────────┘
                         │                          │
              ╔═══════════════════════════════════════════════╗
              ║       WTTransformerLayer × n_layers           ║
              ║  ┌───────────────────────────────────────┐    ║
              ║  │        WTMultiHeadAttention            │    ║
              ║  │                                        │    ║
              ║  │  Q,K,V = Linear(H)(input)              │    ║
              ║  │  A = Q·Kᵀ/√d                          │    ║
              ║  │  A += causal_mask                      │    ║
              ║  │  A += α · wt_bias.permute().unsqueeze()│◄── wt_bias
              ║  │       ★ ATTENTION SCORE MODIFICATION ★ │    ║
              ║  │  out = softmax(A) · V                  │    ║
              ║  │  out = LayerNorm(dense(out) + input)   │    ║
              ║  └────────────────┬──────────────────────┘    ║
              ║                   │                            ║
              ║  ┌────────────────▼──────────────────────┐    ║
              ║  │           FeedForward                  │    ║
              ║  │  FFN(x) = LayerNorm(Linear(Linear(x))) │    ║
              ║  └────────────────┬──────────────────────┘    ║
              ╚═══════════════════╪═══════════════════════════╝
                                  │  (B,L,H)
                         gather last position
                                  │  (B,H)
                          ┌───────┴───────┐
                          │  BPR Loss     │
                          │  pos · seq    │
                          │  neg · seq    │
                          └───────────────┘
```

---

## 4. Learnable Parameters (các tham số học được)

| Parameter | Shape | Defined in | Vai trò |
|---|---|---|---|
| `wt_proj.weight` | `(H_heads, 1)` | `WTSASRec` | Project scalar wt → per-head bias |
| `wt_alpha` | `(1,)` per layer | `WTMultiHeadAttention` | Scale toàn bộ watch-time bias |

Với `n_layers=2, n_heads=2`:
- `wt_proj`: 2 tham số (dùng chung cho tất cả layers)
- `wt_alpha`: 2 tham số (mỗi layer 1 cái)
- **Tổng cộng: 4 tham số bổ sung** trên nền SASRec

---

## 5. Đánh giá tính đúng đắn (Correctness Review)

### ✅ Đúng

| Điểm kiểm tra | Kết quả |
|---|---|
| Bias được thêm vào **trước** softmax | ✅ dòng 63-69 trong `WTMultiHeadAttention.forward` |
| Shape `(B,H,1,L)` broadcast đúng lên `(B,H,L,L)` | ✅ `.permute(0,2,1).unsqueeze(2)` |
| Causal mask vẫn được áp dụng trước bias | ✅ thứ tự: mask trước, wt_bias sau |
| `wt_norm` nằm trong `[0,1]` | ✅ log-normalize per sequence |
| `wt_alpha` learnable (không hardcode) | ✅ `nn.Parameter(torch.ones(1))` |
| Không làm thay đổi non-WT data path (`wt_seq=None`) | ✅ `if wt_bias is not None` check |

### ⚠️ Vấn đề cần lưu ý

#### 1. Dead Code — `WTMultiHeadAttention.wt_proj`
```python
# Trong WTMultiHeadAttention.__init__:
self.wt_proj = nn.Linear(1, n_heads)   # ← KHÔNG BAO GIỜ ĐƯỢC GỌI
```
`wt_proj` cũng được định nghĩa trong `WTSASRec` và được dùng trong `_compute_wt_bias()`.
`wt_proj` trong mỗi `WTMultiHeadAttention` là dead code — tạo ra `n_layers × n_heads` tham số thừa không tham gia huấn luyện hiệu quả.

**Fix:** Xóa `self.wt_proj` khỏi `WTMultiHeadAttention.__init__`.

#### 2. Shared Projection Across Layers
```python
# WTSASRec._compute_wt_bias() tính bias một lần:
wt_bias = self.wt_proj(wt_norm.unsqueeze(-1))  # (B, L, H_heads)

# Sau đó truyền bias NÀY cho TẤT CẢ layers:
for layer_module in self.layer:
    hidden_states = layer_module(hidden_states, attention_mask, wt_bias)  # ← same tensor
```
Tất cả `n_layers` transformer blocks đều nhận cùng một `wt_bias`. Mỗi layer có `wt_alpha` riêng để scale, nhưng cùng projection weights. Điều này đơn giản hơn nhưng ít biểu đạt hơn so với per-layer projection.

**Không ảnh hưởng tính đúng đắn** — đây là lựa chọn thiết kế, có thể tùy chọn nâng cấp.

---

## 6. So sánh với SASRec gốc (Comparison)

```
SASRec gốc:                        WTSASRec:
─────────────────────────────      ────────────────────────────────────
attention_scores = QKᵀ/√d         attention_scores = QKᵀ/√d
                 + mask                             + mask
                                                    + α·wt_bias  ← NEW
probs = softmax(scores)            probs = softmax(scores)
output = probs · V                 output = probs · V
```

Cả hai đều dùng cùng cấu trúc, cùng training loop, cùng evaluation — chỉ khác ở **1 phép cộng** trong attention.

---

## 7. Kết luận

**Có**, WTSASRec **đúng** là triển khai khái niệm **"Sửa đổi attention score"**:

- Watch time được normalize → project thành bias vector per attention head
- Bias được cộng vào attention score **trước softmax** → ảnh hưởng trực tiếp đến phân phối attention
- Item được xem lâu hơn → bias dương cao hơn → attention weight cao hơn → đóng góp nhiều hơn vào prediction

Đây là **minimal-invasive modification**: không thay đổi kiến trúc cơ bản của SASRec, chỉ thêm 1 số tham số nhỏ và 1 phép cộng trong attention computation.
