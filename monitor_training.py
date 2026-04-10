"""
monitor_training.py
-------------------
Real-time training monitor for SASRec / WTSASRec.

Usage:
    python3 monitor_training.py                     # auto-detect latest model log
    python3 monitor_training.py --model SASRec      # watch specific model
    python3 monitor_training.py --refresh 2         # refresh interval in seconds

Shows:
    - GPU: utilization %, VRAM, power draw, temperature
    - Current epoch batch progress bar + live loss
    - Training history table (all completed epochs)
    - ETA per epoch and total

Data sources:
    - training_progress.jsonl  (per-batch stats, written by patched RecBole trainer)
    - log/<MODEL>/*.log        (per-epoch metrics, written by RecBole logger)
"""

import argparse
import glob
import json
import os
import re
import subprocess
import time
from collections import deque
from datetime import datetime, timedelta

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich import box


# ── GPU Stats (via nvidia-smi subprocess) ─────────────────────────────────────

def get_gpu_stats():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "util":    float(parts[0]),
                "mem_used": float(parts[1]),
                "mem_total": float(parts[2]),
                "power":   float(parts[3]),
                "temp":    float(parts[4]),
                "name":    parts[5],
            }
    except Exception:
        pass
    return None


# ── Log Parsing ───────────────────────────────────────────────────────────────

EPOCH_TRAIN_RE = re.compile(
    r"epoch\s+(\d+)\s+training\s+\[time:\s*([\d.]+)s,\s*train loss:\s*([\d.]+)\]"
)
EPOCH_VALID_RE = re.compile(
    r"epoch\s+(\d+)\s+evaluating.*valid_score:\s*([\d.]+)"
)
VALID_RESULT_RE = re.compile(
    r"valid result:.*?ndcg@10\s*:\s*([\d.]+).*?recall@10\s*:\s*([\d.]+).*?mrr@10\s*:\s*([\d.]+)",
    re.DOTALL,
)
TEST_RESULT_RE = re.compile(
    r"test result.*?ndcg@10\s*:\s*([\d.]+).*?recall@10\s*:\s*([\d.]+).*?mrr@10\s*:\s*([\d.]+)",
    re.DOTALL,
)


def find_latest_log(model_name=None):
    pattern = f"log/{model_name}/*.log" if model_name else "log/*/*.log"
    logs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return logs[0] if logs else None


def parse_log(log_path):
    if not log_path or not os.path.exists(log_path):
        return [], None
    
    with open(log_path, errors="replace") as f:
        content = f.read()

    epochs = {}
    for m in EPOCH_TRAIN_RE.finditer(content):
        ep = int(m.group(1))
        epochs.setdefault(ep, {})["train_time"] = float(m.group(2))
        epochs[ep]["train_loss"] = float(m.group(3))

    for m in EPOCH_VALID_RE.finditer(content):
        ep = int(m.group(1))
        epochs.setdefault(ep, {})["valid_score"] = float(m.group(2))

    # Parse valid result blocks per epoch more carefully
    lines = content.split("\n")
    cur_epoch = None
    for i, line in enumerate(lines):
        m = EPOCH_TRAIN_RE.search(line)
        if m:
            cur_epoch = int(m.group(1))
        if "valid result:" in line and cur_epoch is not None:
            block = " ".join(lines[i:i+3])
            for metric, pattern in [
                ("ndcg10", r"ndcg@10\s*:\s*([\d.]+)"),
                ("recall10", r"recall@10\s*:\s*([\d.]+)"),
                ("mrr10", r"mrr@10\s*:\s*([\d.]+)"),
                ("ndcg5", r"ndcg@5\s*:\s*([\d.]+)"),
                ("recall5", r"recall@5\s*:\s*([\d.]+)"),
            ]:
                mm = re.search(pattern, block)
                if mm:
                    epochs.setdefault(cur_epoch, {})[metric] = float(mm.group(1))

    # Parse best valid epoch
    best_epoch = None
    best_m = re.search(r"best valid.*?epoch\s+(\d+)", content, re.IGNORECASE)
    if best_m:
        best_epoch = int(best_m.group(1))

    # Extract model name from log filename
    model_from_log = None
    if log_path:
        parts = log_path.split("/")
        if len(parts) >= 2:
            model_from_log = parts[-2]

    history = [{"epoch": ep, **data} for ep, data in sorted(epochs.items())]
    return history, model_from_log


# ── Batch Progress Parsing ─────────────────────────────────────────────────────

def read_batch_progress(jsonl_path="training_progress.jsonl"):
    if not os.path.exists(jsonl_path):
        return None
    try:
        lines = []
        with open(jsonl_path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        if not lines:
            return None
        last = json.loads(lines[-1])
        # Rolling average of last 50 batches
        recent = []
        for line in lines[-50:]:
            try:
                d = json.loads(line)
                if d.get("epoch") == last.get("epoch"):
                    recent.append(d["loss"])
            except Exception:
                pass
        last["avg_loss"] = sum(recent) / len(recent) if recent else last["loss"]
        last["recent_losses"] = recent
        return last
    except Exception:
        return None


# ── UI Components ─────────────────────────────────────────────────────────────

def make_gpu_panel(stats):
    if stats is None:
        return Panel("[yellow]nvidia-smi not available[/yellow]", title="🖥️  GPU", border_style="yellow")
    
    util_color = "green" if stats["util"] < 80 else ("yellow" if stats["util"] < 95 else "red")
    temp_color = "green" if stats["temp"] < 70 else ("yellow" if stats["temp"] < 85 else "red")
    mem_pct = stats["mem_used"] / stats["mem_total"] * 100
    mem_color = "green" if mem_pct < 70 else ("yellow" if mem_pct < 90 else "red")
    power_pct = stats["power"] / 250 * 100  # RTX 3090 TDP = 250W

    # Build utilization bar
    def bar(pct, width=20):
        filled = int(pct / 100 * width)
        return f"[{'#' * filled}{'-' * (width - filled)}]"

    lines = [
        f"[bold]{stats['name']}[/bold]",
        f"  Utilization : [{util_color}]{stats['util']:5.1f}%[/] {bar(stats['util'])}",
        f"  VRAM        : [{mem_color}]{stats['mem_used']:6.0f} / {stats['mem_total']:.0f} MB[/] ({mem_pct:.1f}%)",
        f"  Power       : [cyan]{stats['power']:5.1f}W[/] / 250W  {bar(power_pct)}",
        f"  Temperature : [{temp_color}]{stats['temp']:.0f}°C[/]",
    ]
    return Panel("\n".join(lines), title="🖥️  GPU", border_style="cyan")


def make_batch_panel(batch_info, history):
    if batch_info is None:
        # Try to estimate from log timing
        msg = "[dim]Waiting for training to start...\n(training_progress.jsonl not yet created)[/dim]"
        return Panel(msg, title="📊 Batch Progress", border_style="blue")

    epoch = batch_info["epoch"]
    batch = batch_info["batch"]
    total = batch_info["total_batches"]
    loss  = batch_info["loss"]
    avg   = batch_info.get("avg_loss", loss)
    ts    = batch_info.get("ts", time.time())

    pct = batch / total if total > 0 else 0
    bar_width = 40
    filled = int(pct * bar_width)
    bar = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_width - filled)}[/dim]"

    # Estimate time per batch from recent timestamps
    elapsed_epoch = time.time() - ts + (batch * 2)  # rough estimate
    eta_batch_s = "--"
    remaining = total - batch
    if batch > 0:
        # Use average of recent losses timestamps if possible
        eta_batch_s = f"~{int(remaining * 0.2)}s" if remaining > 0 else "done"

    lines = [
        f"  Epoch [bold cyan]{epoch}[/bold cyan]  |  Batch [bold]{batch}/{total}[/bold]  ({pct*100:.1f}%)",
        f"  {bar}",
        f"  Current Loss : [yellow]{loss:.4f}[/yellow]   Avg (last 50) : [green]{avg:.4f}[/green]",
    ]

    # Recent loss sparkline (ASCII)
    recent = batch_info.get("recent_losses", [])
    if len(recent) > 2:
        mn, mx = min(recent), max(recent)
        spark_chars = "▁▂▃▄▅▆▇█"
        sparkline = ""
        for v in recent[-30:]:
            idx = int((v - mn) / (mx - mn + 1e-8) * 7)
            sparkline += spark_chars[idx]
        lines.append(f"  Loss trend   : [dim]{sparkline}[/dim]")

    return Panel("\n".join(lines), title="📊 Batch Progress", border_style="blue")


def make_history_table(history, model_name=""):
    table = Table(
        title=f"📈 Epoch History — {model_name}",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Epoch", style="cyan",  justify="right", width=7)
    table.add_column("Train Loss", justify="right", width=12)
    table.add_column("NDCG@10",    justify="right", width=10)
    table.add_column("Recall@10",  justify="right", width=10)
    table.add_column("MRR@10",     justify="right", width=10)
    table.add_column("Val Score",  justify="right", width=10)
    table.add_column("Time",       justify="right", width=8)

    if not history:
        table.add_row("[dim]No epochs completed yet[/dim]", *["—"] * 6)
        return table

    for row in history[-20:]:  # show last 20 epochs
        ep          = str(row["epoch"])
        train_loss  = f"{row.get('train_loss', 0):.4f}" if "train_loss" in row else "—"
        ndcg10      = f"[green]{row['ndcg10']:.4f}[/green]" if "ndcg10" in row else "—"
        recall10    = f"{row.get('recall10', 0):.4f}" if "recall10" in row else "—"
        mrr10       = f"{row.get('mrr10', 0):.4f}" if "mrr10" in row else "—"
        valid_score = f"{row.get('valid_score', 0):.4f}" if "valid_score" in row else "—"
        t           = f"{row.get('train_time', 0):.0f}s" if "train_time" in row else "—"
        table.add_row(ep, train_loss, ndcg10, recall10, mrr10, valid_score, t)

    return table


def make_eta_panel(history, batch_info):
    if not history:
        return Panel("[dim]Waiting for first epoch...[/dim]", title="⏱️  ETA", border_style="dim")

    # Estimate epoch time from history
    times = [r["train_time"] for r in history if "train_time" in r]
    avg_epoch_time = sum(times) / len(times) if times else 270  # default ~4.5min

    # How many epochs done
    epochs_done = len(history)
    total_epochs = 50  # config max

    # Early stopping: assume stops ~epoch 20-30 based on trend
    # Check if NDCG is still improving
    ndcg_vals = [r.get("ndcg10", 0) for r in history if "ndcg10" in r]
    patience_left = 10
    if len(ndcg_vals) >= 2:
        best_idx = ndcg_vals.index(max(ndcg_vals))
        patience_left = max(0, 10 - (len(ndcg_vals) - 1 - best_idx))

    remaining_epochs = min(patience_left, total_epochs - epochs_done)
    eta_total = timedelta(seconds=int(remaining_epochs * avg_epoch_time))
    eta_epoch = timedelta(seconds=int(avg_epoch_time))

    best_ndcg = max(ndcg_vals) if ndcg_vals else 0
    best_epoch_idx = ndcg_vals.index(best_ndcg) if ndcg_vals else 0

    lines = [
        f"  Epochs done  : [cyan]{epochs_done}[/cyan] / {total_epochs} (early-stop patience: {patience_left}/10)",
        f"  Avg epoch    : [yellow]{avg_epoch_time:.0f}s[/yellow] (~{avg_epoch_time/60:.1f} min)",
        f"  ETA total    : [bold green]{str(eta_total)}[/bold green]  (~{remaining_epochs} epochs left)",
        f"  Best NDCG@10 : [bold magenta]{best_ndcg:.4f}[/bold magenta] (epoch {history[best_epoch_idx]['epoch']})",
    ]
    return Panel("\n".join(lines), title="⏱️  ETA", border_style="green")


def make_header(model_name, log_path, refresh_interval):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return Panel(
        f"[bold cyan]🚀 SASRec / WTSASRec Training Monitor[/bold cyan]   "
        f"[dim]Model: [bold]{model_name or 'auto-detect'}[/bold]   "
        f"Log: {log_path or 'searching...'}   "
        f"Refresh: {refresh_interval}s   "
        f"Now: {now}[/dim]",
        border_style="bright_blue",
    )


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time training monitor")
    parser.add_argument("--model",   default=None, help="Model name (SASRec, WTSASRec)")
    parser.add_argument("--refresh", default=2, type=float, help="Refresh interval in seconds")
    parser.add_argument("--log",     default=None, help="Path to specific log file")
    args = parser.parse_args()

    console = Console()
    cwd = os.path.dirname(os.path.abspath(__file__))
    os.chdir(cwd)
    jsonl_path = os.path.join(cwd, "training_progress.jsonl")

    console.print(f"[bold cyan]Monitor started[/bold cyan] — watching [yellow]{cwd}[/yellow]")
    console.print(f"  Log dir      : [dim]{cwd}/log/[/dim]")
    console.print(f"  Batch file   : [dim]{jsonl_path}[/dim]")
    console.print(f"  Refresh      : {args.refresh}s\n")
    console.print("[dim]Press Ctrl+C to exit[/dim]\n")
    time.sleep(1)

    with Live(console=console, refresh_per_second=1, screen=False) as live:
        while True:
            try:
                # Find latest log
                log_path = args.log or find_latest_log(args.model)
                history, model_from_log = parse_log(log_path)
                model_name = args.model or model_from_log or "unknown"

                # GPU stats
                gpu_stats = get_gpu_stats()

                # Batch progress
                batch_info = read_batch_progress(jsonl_path)

                # Build layout
                layout = Layout()
                layout.split_column(
                    Layout(make_header(model_name, log_path, args.refresh), size=3),
                    Layout(name="top", size=9),
                    Layout(make_history_table(history, model_name), size=min(len(history) + 5, 28)),
                    Layout(make_eta_panel(history, batch_info), size=7),
                )
                layout["top"].split_row(
                    Layout(make_gpu_panel(gpu_stats), ratio=1),
                    Layout(make_batch_panel(batch_info, history), ratio=2),
                )

                live.update(layout)
                time.sleep(args.refresh)

            except KeyboardInterrupt:
                console.print("\n[bold red]Monitor stopped.[/bold red]")
                break
            except Exception as e:
                console.print(f"[red]Monitor error: {e}[/red]")
                time.sleep(args.refresh)


if __name__ == "__main__":
    main()
