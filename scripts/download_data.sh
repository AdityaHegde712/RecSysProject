#!/bin/bash
# Download HotelRec dataset
#
# The dataset is hosted on SWITCHdrive (Swiss academic cloud):
#   https://drive.switch.ch/index.php/s/n48smsdufhRA7fR
#
# It's a single large JSON file (~10GB) containing all 50M reviews.
# For academic research use only.
#
# Usage:
#   bash scripts/download_data.sh full     # download from SWITCHdrive
#   bash scripts/download_data.sh sample   # generate synthetic data for testing
#   bash scripts/download_data.sh          # defaults to sample

set -o pipefail

MODE="${1:-sample}"
RAW_DIR="data/raw"

mkdir -p "$RAW_DIR"

echo "============================================"
echo "HotelRec Dataset Download"
echo "============================================"
echo "Mode: $MODE"
echo "Target: $RAW_DIR/"
echo "Date: $(date)"
echo "============================================"
echo ""

# ─── Full download from SWITCHdrive ──────────────────────────────────
download_full() {
    echo ">>> Downloading HotelRec dataset from SWITCHdrive..."
    echo "    Source: https://drive.switch.ch/index.php/s/n48smsdufhRA7fR"
    echo "    The archive is ~10GB (zip containing ~365K hotel JSON files)."
    echo "    No extraction needed — our pipeline streams directly from the zip."
    echo ""

    DOWNLOAD_URL="https://drive.switch.ch/index.php/s/n48smsdufhRA7fR/download"

    # Check if we already have the data (zip or extracted JSON files)
    EXISTING_ZIP=$(ls "${RAW_DIR}"/*.zip 2>/dev/null | head -1)
    EXISTING_JSON=$(ls "${RAW_DIR}"/*.json 2>/dev/null | head -1)

    if [ -n "$EXISTING_ZIP" ]; then
        SIZE=$(du -h "$EXISTING_ZIP" | cut -f1)
        echo "  Archive already exists: $EXISTING_ZIP ($SIZE)"
        echo "  Delete it first if you want to re-download."
        return 0
    fi

    if [ -n "$EXISTING_JSON" ]; then
        N_JSON=$(ls "${RAW_DIR}"/*.json 2>/dev/null | wc -l | tr -d ' ')
        echo "  Found $N_JSON JSON files already in ${RAW_DIR}/"
        echo "  Looks like data is already extracted. Skipping download."
        return 0
    fi

    DEST="${RAW_DIR}/HotelRec.zip"
    echo "  Downloading to $DEST ..."
    echo "  (This will take a while — ~10GB file)"
    echo ""

    # Try wget first (supports resume with -c)
    if command -v wget &> /dev/null; then
        wget -c -O "$DEST" "$DOWNLOAD_URL" || {
            echo ""
            echo "ERROR: wget failed. Try manually:"
            echo "  wget -c -O $DEST '$DOWNLOAD_URL'"
            exit 1
        }
    elif command -v curl &> /dev/null; then
        curl -L -C - -o "$DEST" "$DOWNLOAD_URL" || {
            echo ""
            echo "ERROR: curl failed. Try manually:"
            echo "  curl -L -C - -o $DEST '$DOWNLOAD_URL'"
            exit 1
        }
    else
        echo "ERROR: Neither wget nor curl found."
        echo "Download manually from: https://drive.switch.ch/index.php/s/n48smsdufhRA7fR"
        echo "Save to: $DEST"
        exit 1
    fi

    # The zip contains ~365K per-hotel JSON files. preprocess.py expects a
    # single JSONL file at data/raw/HotelRec.txt (one review per line).
    # We stream from the zip and flatten into JSONL — avoids keeping both
    # the zip and extracted file on disk simultaneously.
    _extract_zip_to_jsonl "${DEST}" "" # empty = all reviews
}

# ─── Shared: stream zip → JSONL ──────────────────────────────────────
_extract_zip_to_jsonl() {
    local ZIP_PATH="$1"
    local MAX_REVIEWS="$2"  # empty string = no limit
    local TXT_FILE="${RAW_DIR}/HotelRec.txt"

    if [ -f "$TXT_FILE" ]; then
        LINES=$(wc -l < "$TXT_FILE" | tr -d ' ')
        echo "  [OK] ${TXT_FILE} already exists (${LINES} reviews) — skipping"
        return 0
    fi

    if [ ! -f "$ZIP_PATH" ]; then
        echo "ERROR: Zip file not found at ${ZIP_PATH}"
        exit 1
    fi

    local LIMIT_MSG="all"
    [ -n "$MAX_REVIEWS" ] && LIMIT_MSG="first ${MAX_REVIEWS}"
    echo "  Streaming ${LIMIT_MSG} reviews from zip → ${TXT_FILE}..."
    echo ""

    python3 -c "
import zipfile, json

zip_path = '${ZIP_PATH}'
out_path = '${TXT_FILE}'
max_reviews = ${MAX_REVIEWS:-0}  # 0 = no limit
count = 0

with zipfile.ZipFile(zip_path, 'r') as zf:
    names = [n for n in zf.namelist() if n.endswith('.json')]
    print(f'  Found {len(names)} JSON files in archive')
    with open(out_path, 'w', encoding='utf-8') as out:
        for i, name in enumerate(names):
            with zf.open(name) as f:
                try:
                    data = json.load(f)
                except Exception:
                    continue
                reviews = data if isinstance(data, list) else [data]
                for review in reviews:
                    out.write(json.dumps(review) + '\n')
                    count += 1
                    if max_reviews > 0 and count >= max_reviews:
                        break
            if max_reviews > 0 and count >= max_reviews:
                break
            if (i + 1) % 10000 == 0:
                print(f'  Processed {i+1}/{len(names)} files, {count:,} reviews so far')

print(f'  Done: {count:,} reviews written to {out_path}')
" || {
        echo "ERROR: Extraction failed!"
        exit 1
    }
}

# ─── Sample: stream first N reviews from zip ─────────────────────────
SAMPLE_REVIEWS=500000  # ~500K reviews for a quick smoke test

download_sample() {
    echo ">>> Creating sample dataset (first ${SAMPLE_REVIEWS} reviews)..."

    # Need the zip file
    ZIP_FILE=$(ls "${RAW_DIR}"/*.zip 2>/dev/null | head -1)
    if [ -z "$ZIP_FILE" ]; then
        echo ""
        echo "ERROR: No zip file found in ${RAW_DIR}/"
        echo "Download the full dataset first:"
        echo "  bash scripts/download_data.sh full"
        echo ""
        echo "Then run sample to extract a subset:"
        echo "  bash scripts/download_data.sh sample"
        exit 1
    fi

    # Remove existing HotelRec.txt so _extract_zip_to_jsonl runs
    rm -f "${RAW_DIR}/HotelRec.txt"

    _extract_zip_to_jsonl "$ZIP_FILE" "$SAMPLE_REVIEWS"
}

# ─── Dispatch ─────────────────────────────────────────────────────────
case "$MODE" in
    full)
        download_full
        ;;
    sample)
        download_sample
        ;;
    *)
        echo "Usage: bash scripts/download_data.sh [full|sample]"
        echo "  full   — download from SWITCHdrive (~10GB)"
        echo "  sample — generate synthetic data for testing"
        exit 1
        ;;
esac

# ─── Stats ────────────────────────────────────────────────────────────
echo ""
echo "--- Dataset stats ---"
TXT_CHECK="${RAW_DIR}/HotelRec.txt"
if [ -f "$TXT_CHECK" ]; then
    SIZE=$(du -h "$TXT_CHECK" | cut -f1)
    LINES=$(wc -l < "$TXT_CHECK" | tr -d ' ')
    echo "  JSONL file: HotelRec.txt ($SIZE, ${LINES} lines)"
else
    ZIP_FILE=$(ls "${RAW_DIR}"/*.zip 2>/dev/null | head -1)
    if [ -n "$ZIP_FILE" ]; then
        SIZE=$(du -h "$ZIP_FILE" | cut -f1)
        echo "  Archive: $(basename "$ZIP_FILE") ($SIZE)"
        echo "  NOTE: Run download again to extract to HotelRec.txt"
    else
        echo "  No data files found in ${RAW_DIR}/"
    fi
fi
echo "---------------------"
