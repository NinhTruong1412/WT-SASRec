"""
Generate WTSASRec architecture flow diagram using matplotlib.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 18))
ax.set_xlim(0, 14)
ax.set_ylim(0, 18)
ax.axis('off')
fig.patch.set_facecolor('#FAFBFF')
ax.set_facecolor('#FAFBFF')

# ── Color palette ──────────────────────────────────────────────────────────────
C_INPUT    = '#2C3E75'   # dark navy  – input boxes
C_NORM     = '#1A6B8A'   # teal       – normalization
C_PROJ     = '#1A6B8A'
C_EMBED    = '#2E7D32'   # dark green – embedding
C_ATTN     = '#6A1B9A'   # purple     – attention (core)
C_WT       = '#C62828'   # red        – watch-time highlight
C_FFN      = '#4527A0'   # indigo     – FFN
C_OUT      = '#1B5E20'   # deep green – output / loss
C_STACK    = '#E8EAF6'   # light lav  – transformer stack bg
C_ARROW    = '#455A64'
C_WTARROW  = '#C62828'

def box(ax, x, y, w, h, label, sublabel=None, color='#2C3E75', fontsize=10,
        radius=0.25, text_color='white', bold=True):
    fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad=0.05,rounding_size={radius}",
                           linewidth=1.5, edgecolor='white',
                           facecolor=color, zorder=3)
    ax.add_patch(fancy)
    weight = 'bold' if bold else 'normal'
    ya = y if sublabel is None else y + h * 0.15
    ax.text(x, ya, label, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=4)
    if sublabel:
        ax.text(x, y - h * 0.22, sublabel, ha='center', va='center',
                fontsize=fontsize - 1.5, color=text_color, alpha=0.88, zorder=4,
                style='italic')

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=2, label=None, label_side='right'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        dx = 0.15 if label_side == 'right' else -0.15
        ax.text(mx + dx, my, label, ha='left' if label_side=='right' else 'right',
                va='center', fontsize=8, color=color, style='italic')

def curved_arrow(ax, x1, y1, x2, y2, rad=0.3, color=C_ARROW, lw=2, label=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'))
    if label:
        mx, my = (x1+x2)/2 + 0.5*rad, (y1+y2)/2
        ax.text(mx, my, label, ha='center', va='center', fontsize=8, color=color)

def section_bg(ax, x, y, w, h, color, label=None, fontsize=8.5):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05,rounding_size=0.3",
                          linewidth=1.5, edgecolor=color,
                          facecolor=color + '18', zorder=1, linestyle='--')
    ax.add_patch(rect)
    if label:
        ax.text(x + w/2, y + h - 0.12, label, ha='center', va='top',
                fontsize=fontsize, color=color, fontweight='bold', zorder=2)

# ── Layout constants ──────────────────────────────────────────────────────────
LEFT_X  = 4.0    # main pipeline center-x
RIGHT_X = 10.5   # watch-time pipeline center-x
BW = 3.6         # box width (main)
BWS = 3.0        # box width (small)
BH = 0.62        # box height

# ── TITLE ─────────────────────────────────────────────────────────────────────
ax.text(7, 17.55, 'WTSASRec Architecture', ha='center', va='center',
        fontsize=17, fontweight='bold', color='#16213e')
ax.text(7, 17.15, 'Watch-Time Augmented Self-Attentive Sequential Recommendation',
        ha='center', va='center', fontsize=10, color='#555', style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# INPUT LAYER (y=16.3)
# ─────────────────────────────────────────────────────────────────────────────
box(ax, LEFT_X,   16.3, BW, BH, 'item_seq  (B, L)', 'Item ID sequence (padded)', C_INPUT, fontsize=9.5)
box(ax, RIGHT_X,  16.3, BWS, BH, 'watch_time_list  (B, L)', 'Raw watch time [seconds]', C_WT, fontsize=9.5)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Normalize  (right branch, y=15.3)
# ─────────────────────────────────────────────────────────────────────────────
arrow(ax, RIGHT_X, 15.99, RIGHT_X, 15.65)
box(ax, RIGHT_X, 15.3, BWS, BH,
    'Log-Normalize  (Step 1)',
    r'w̃ = log(1+wt) / max(log(1+wt))',
    C_NORM, fontsize=9)

# STEP 2 — Project (right branch, y=14.3)
arrow(ax, RIGHT_X, 14.99, RIGHT_X, 14.65)
box(ax, RIGHT_X, 14.3, BWS, BH,
    'Linear Projection  (Step 2)',
    r'b_wt = W_proj · w̃  →  (B, L, H_heads)',
    C_PROJ, fontsize=9)

# Reshape annotation
arrow(ax, RIGHT_X, 13.99, RIGHT_X, 13.72)
ax.text(RIGHT_X, 13.57,
        'reshape: (B, H_heads, 1, L_k)', ha='center', va='center',
        fontsize=8.2, color=C_WT, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor=C_WT, lw=1))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Embedding  (left branch, y=15.3)
# ─────────────────────────────────────────────────────────────────────────────
arrow(ax, LEFT_X, 15.99, LEFT_X, 15.65)
box(ax, LEFT_X, 15.3, BW, BH,
    'Item + Position Embedding  (Step 3)',
    'LayerNorm( Dropout( e_item + e_pos ) )  →  (B, L, H)',
    C_EMBED, fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMER STACK BACKGROUND  (y=10.8 to 13.4)
# ─────────────────────────────────────────────────────────────────────────────
section_bg(ax, 0.9, 10.7, 9.8, 3.1, '#6A1B9A', '× n_layers  (Transformer Stack)', fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — WTMultiHeadAttention  (y=13.05)
# ─────────────────────────────────────────────────────────────────────────────
arrow(ax, LEFT_X, 14.99, LEFT_X, 13.38)
box(ax, LEFT_X, 13.05, BW + 0.4, 0.72,
    'WTMultiHeadAttention  (Step 4)', None, C_ATTN, fontsize=10)

# Sub-steps inside attention block
SUB_X = 5.7
sub_items = [
    (12.52, 'Q, K, V  =  Linear(H)(input_emb)',               '#7B1FA2'),
    (11.97, 'A  =  Q·Kᵀ / √d',                                '#7B1FA2'),
    (11.42, 'A  +=  causal_mask        (future = −∞)',          '#5C6BC0'),
    (10.87, 'A  +=  α · b_wt           ★ WT MODIFICATION',     C_WT),
    (10.32, 'probs  =  softmax(A);   output  =  probs · V',    '#7B1FA2'),
]
for sy, slabel, sc in sub_items:
    lw = 2.0 if sc == C_WT else 1.2
    fancy = FancyBboxPatch((1.3, sy - 0.22), 8.5, 0.42,
                           boxstyle="round,pad=0.04,rounding_size=0.12",
                           linewidth=lw, edgecolor=sc,
                           facecolor=sc + '22', zorder=3)
    ax.add_patch(fancy)
    fw = 'bold' if sc == C_WT else 'normal'
    ax.text(SUB_X, sy, slabel, ha='center', va='center',
            fontsize=9, color=sc, fontweight=fw, zorder=4)

# WT bias arrow from right branch into attention
arrow(ax, RIGHT_X - 1.5, 13.57, 9.85, 10.87,
      color=C_WT, lw=2.5, label='b_wt', label_side='right')

# Residual connection note
ax.text(LEFT_X, 10.05, 'LayerNorm( dense(output) + input_emb )  — residual',
        ha='center', va='center', fontsize=8.5, color='#4527A0',
        bbox=dict(boxstyle='round,pad=0.28', facecolor='#EDE7F6', edgecolor='#4527A0', lw=1))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FeedForward  (y=9.25)
# ─────────────────────────────────────────────────────────────────────────────
arrow(ax, LEFT_X, 10.7, LEFT_X, 9.59)
box(ax, LEFT_X, 9.25, BW, BH,
    'FeedForward  (Step 5)',
    'LayerNorm( FFN(x) + x )  →  (B, L, H)',
    C_FFN, fontsize=9)

# Close stack bracket
arrow(ax, LEFT_X, 8.94, LEFT_X, 8.63)
ax.text(LEFT_X, 8.47,
        '↑ Repeat × n_layers  |  same b_wt, per-layer α',
        ha='center', va='center', fontsize=8.2, color='#6A1B9A',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#EDE7F6', edgecolor='#6A1B9A', lw=1))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Gather last position  (y=7.5)
# ─────────────────────────────────────────────────────────────────────────────
arrow(ax, LEFT_X, 8.2, LEFT_X, 7.85)
box(ax, LEFT_X, 7.55, BW, BH,
    'Gather Last Real Position  (Step 6)',
    'seq_repr  =  output[:, seq_len−1, :]  →  (B, H)',
    C_OUT, fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# BPR LOSS  (y=6.4)
# ─────────────────────────────────────────────────────────────────────────────
arrow(ax, LEFT_X, 7.24, LEFT_X, 6.76)
box(ax, LEFT_X, 6.43, BW, BH,
    'BPR Loss',
    'loss  =  −log σ( score_pos − score_neg )',
    C_OUT, fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# FORMULA SUMMARY BOX  (y=5.4)
# ─────────────────────────────────────────────────────────────────────────────
arrow(ax, LEFT_X, 6.12, LEFT_X, 5.72)

summary_bg = FancyBboxPatch((1.2, 4.55), 11.6, 1.05,
                             boxstyle="round,pad=0.05,rounding_size=0.3",
                             linewidth=2, edgecolor='#0f3460',
                             facecolor='#E3F2FD', zorder=3)
ax.add_patch(summary_bg)
ax.text(7, 5.3, 'Core Formula:',
        ha='center', va='center', fontsize=9.5, color='#0f3460',
        fontweight='bold', zorder=4)
ax.text(7, 4.9,
        r'A  =  QKᵀ/√d  +  M  +  α · W_proj · w̃       →       softmax(A) · V',
        ha='center', va='center', fontsize=10.5, color='#C62828',
        fontweight='bold', zorder=4,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#C62828', lw=1.2))

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND  (y=3.9 → 2.2)
# ─────────────────────────────────────────────────────────────────────────────
leg_y = 4.0
ax.text(7, leg_y, 'Legend', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#333')

legend_items = [
    (C_INPUT,  'Input data'),
    (C_EMBED,  'Embedding layer'),
    (C_NORM,   'Watch-time normalization & projection'),
    (C_ATTN,   'Self-attention (modified)'),
    (C_WT,     '★ Watch-time modification  (A += α · b_wt)'),
    (C_FFN,    'Feed-forward network'),
    (C_OUT,    'Output / Loss'),
]
for i, (lc, ltxt) in enumerate(legend_items):
    row = i // 2
    col = i % 2
    lx = 1.8 + col * 6.5
    ly = leg_y - 0.55 - row * 0.52
    rect = FancyBboxPatch((lx, ly - 0.15), 0.45, 0.3,
                          boxstyle="round,pad=0.04,rounding_size=0.06",
                          facecolor=lc, edgecolor='white', lw=1, zorder=3)
    ax.add_patch(rect)
    fw = 'bold' if lc == C_WT else 'normal'
    ax.text(lx + 0.62, ly, ltxt, va='center', fontsize=8.5,
            color='#222', fontweight=fw)

# ─────────────────────────────────────────────────────────────────────────────
# STEP labels on the right margin
# ─────────────────────────────────────────────────────────────────────────────
step_labels = [
    (16.3, 'INPUT'),
    (15.3, 'STEP 1'),
    (14.3, 'STEP 2'),
    (15.3 - 0.02, 'STEP 3'),   # same y as norm but left branch
    (13.05, 'STEP 4'),
    (9.25, 'STEP 5'),
    (7.55, 'STEP 6'),
]
# Only label main-pipeline steps on the left
for sy, slbl in [(16.3,'INPUT'),(15.3,'STEP 3'),(13.05,'STEP 4'),
                 (9.25,'STEP 5'),(7.55,'STEP 6'),(6.43,'LOSS')]:
    ax.text(0.55, sy, slbl, ha='center', va='center',
            fontsize=7.5, color='#888', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#EEE', edgecolor='#CCC', lw=0.8))

# Watch-time branch labels
for sy, slbl in [(16.3,'INPUT'),(15.3,'STEP 1'),(14.3,'STEP 2')]:
    ax.text(13.55, sy, slbl, ha='center', va='center',
            fontsize=7.5, color='#888', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#EEE', edgecolor='#CCC', lw=0.8))

plt.tight_layout(pad=0.3)
out = '/Users/ninh.truong/Desktop/Master/Sequential Model/docs/wt_sasrec_diagram.png'
plt.savefig(out, dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none')
print(f"Saved: {out}")
plt.close()
