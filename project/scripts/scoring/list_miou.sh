#!/bin/bash
cd ~/work/results
grep -rl "mIoU of 6019" . | sort | while read f; do
  echo "$f: $(grep -o 'mIoU of 6019 samples: [0-9.]*' "$f" | tail -1)"
done
