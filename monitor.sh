#!/bin/bash
# Monitor training and signal when pipeline is complete
cd /workspace/master/WT-SASRec
while true; do
    if grep -q "Pipeline complete!" pipeline.log 2>/dev/null; then
        echo "DONE"
        break
    fi
    sleep 60
    tail -3 pipeline.log 2>/dev/null
done
