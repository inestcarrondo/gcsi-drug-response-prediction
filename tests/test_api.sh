#!/usr/bin/env bash
set -euo pipefail

echo "Health:"
curl -f -s http://localhost:8000/health

echo "Predict:"
curl -f -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "samples": [
          {"norm_drugname":"VINBLASTINE","norm_celllinename":"SKBR3","trtduration":72,"doublingtime":73.0,"maxlog10concentration":-7.0,"primarytissue":"Breast"}
        ]
      }'
echo "OK"
