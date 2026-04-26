#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash submit.sh
#   bash submit.sh my_submission.zip
OUT_ZIP="${1:-assignment4_submission.zip}"
if [[ "${OUT_ZIP}" != *.zip ]]; then
  OUT_ZIP="${OUT_ZIP}.zip"
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: 'zip' is not installed. Please install it and rerun."
  exit 1
fi

rm -f "${OUT_ZIP}"

# Package all assignment files except data, cache, and zip files.
# If it doesn't work, zip scripts/ and the notebook manually,
# and make sure to exclude the data/ folder.
zip -r "${OUT_ZIP}" . \
  -x "data/*" \
     "*/data/*" \
     "proj5_code/*" \
     "__pycache__/*" \
     "*/__pycache__/*" \
     "*.pyc" \
     ".DS_Store" \
     "*.zip" \
     "*/.ipynb_checkpoints/*" \
     "*.yml" \
     "*.tflite" \
     "*.task" \
     "assignment4_instructor/*"

echo "Created ${OUT_ZIP}"
