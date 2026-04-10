#!/bin/bash
# Pipeline: Train SASRec → WTSASRec → Generate report
set -e
cd /workspace/master/WT-SASRec
LOG=/workspace/master/WT-SASRec/pipeline.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Pipeline started ==="
log "Step 1: Training SASRec (full dataset, GPU)..."
python3 run_baseline.py --model SASRec 2>&1 | tee -a "$LOG"
log "Step 1 DONE: SASRec training complete."

log "Step 2: Training WTSASRec (full dataset, GPU)..."
python3 run_baseline.py --model WTSASRec --weighted 2>&1 | tee -a "$LOG"
log "Step 2 DONE: WTSASRec training complete."

log "Step 3: Generating comparison report..."
python3 generate_report.py 2>&1 | tee -a "$LOG"
log "=== Pipeline complete! ==="
