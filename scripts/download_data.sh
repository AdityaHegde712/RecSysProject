#!/bin/bash
# Download HotelRec dataset from GitHub
#
# The HotelRec dataset contains ~50M TripAdvisor hotel reviews stored as
# JSON files (one per hotel). The full dataset is ~10GB compressed.
#
# Usage:
#   bash scripts/download_data.sh full     # download entire dataset
#   bash scripts/download_data.sh sample   # download a small subset for testing
#   bash scripts/download_data.sh          # defaults to sample
#
# Run this on the LOGIN NODE (has internet access).

set -o pipefail

MODE="${1:-sample}"
RAW_DIR="data/raw"
REPO_URL="https://github.com/Diego999/HotelRec"

mkdir -p "$RAW_DIR"

echo "============================================"
echo "HotelRec Dataset Download"
echo "============================================"
echo "Mode: $MODE"
echo "Target: $RAW_DIR/"
echo "Date: $(date)"
echo "============================================"
echo ""

# ─── Full download ────────────────────────────────────────────────────
download_full() {
    echo ">>> Downloading full HotelRec dataset..."
    echo "    This is ~10GB compressed, ~50GB uncompressed."
    echo ""

    CLONE_DIR="${RAW_DIR}/.hotelrec_repo"

    if [ -d "$CLONE_DIR" ]; then
        echo "Repository already cloned at ${CLONE_DIR}"
    else
        echo "Cloning HotelRec repository (metadata only)..."
        git clone --depth 1 "$REPO_URL" "$CLONE_DIR" 2>&1 || {
            echo "ERROR: git clone failed."
            echo "Try downloading manually from: ${REPO_URL}"
            exit 1
        }
    fi

    # The dataset files are typically distributed via GitHub releases
    # or external links referenced in the repo README.
    # Check for release assets first.
    echo ""
    echo "Checking for dataset download links..."

    # Try GitHub releases API
    RELEASES_URL="https://api.github.com/repos/Diego999/HotelRec/releases"
    RELEASE_INFO=$(curl -s "$RELEASES_URL" 2>/dev/null | head -200)

    if echo "$RELEASE_INFO" | grep -q "browser_download_url"; then
        echo "Found release assets. Downloading..."
        DOWNLOAD_URLS=$(echo "$RELEASE_INFO" | grep -o '"browser_download_url": "[^"]*"' | cut -d'"' -f4)
        for url in $DOWNLOAD_URLS; do
            fname=$(basename "$url")
            dest="${RAW_DIR}/${fname}"
            if [ -f "$dest" ]; then
                echo "  ${fname}: already exists — skipping"
            else
                echo "  Downloading ${fname}..."
                curl -L -o "$dest" "$url" || echo "  WARNING: failed to download ${fname}"
            fi
        done
    else
        echo "No release assets found. Checking repo for data files..."

        # Look for JSON files or archives in the cloned repo
        if ls "${CLONE_DIR}"/*.json 2>/dev/null | head -1 > /dev/null; then
            echo "Found JSON files in repo. Copying to ${RAW_DIR}/..."
            cp "${CLONE_DIR}"/*.json "$RAW_DIR/" 2>/dev/null || true
        elif ls "${CLONE_DIR}"/data/*.json 2>/dev/null | head -1 > /dev/null; then
            echo "Found JSON files in repo/data/. Copying..."
            cp "${CLONE_DIR}"/data/*.json "$RAW_DIR/" 2>/dev/null || true
        fi

        # Check for compressed archives
        for ext in tar.gz tar.bz2 zip; do
            for archive in "${CLONE_DIR}"/*."${ext}" "${CLONE_DIR}"/data/*."${ext}"; do
                if [ -f "$archive" ]; then
                    echo "Extracting $(basename "$archive")..."
                    case "$ext" in
                        tar.gz)  tar xzf "$archive" -C "$RAW_DIR/" ;;
                        tar.bz2) tar xjf "$archive" -C "$RAW_DIR/" ;;
                        zip)     unzip -o "$archive" -d "$RAW_DIR/" ;;
                    esac
                fi
            done
        done
    fi

    # Check README for external download links (e.g., Zenodo, Google Drive)
    if [ -f "${CLONE_DIR}/README.md" ]; then
        echo ""
        echo "--- README download instructions ---"
        grep -i -A2 "download\|dataset\|data.*link\|zenodo\|drive.google" "${CLONE_DIR}/README.md" 2>/dev/null | head -20
        echo "------------------------------------"
    fi
}

# ─── Sample download ─────────────────────────────────────────────────
download_sample() {
    echo ">>> Creating sample dataset for development/testing..."
    echo "    Generating synthetic hotel review data."
    echo ""

    SAMPLE_DIR="${RAW_DIR}"
    mkdir -p "$SAMPLE_DIR"

    # Generate a small synthetic dataset that matches the HotelRec JSON format.
    # This lets us test the full pipeline without downloading 10GB.
    python3 -c "
import json
import random
import os

random.seed(42)
raw_dir = '${SAMPLE_DIR}'

# generate 20 hotels with 50-200 reviews each
num_hotels = 20
user_pool = [f'user_{i}' for i in range(200)]

total_reviews = 0
for h in range(num_hotels):
    hotel_url = f'hotel_{h:04d}'
    n_reviews = random.randint(50, 200)
    reviews = []
    for r in range(n_reviews):
        user = random.choice(user_pool)
        overall = random.randint(1, 5)
        review = {
            'user_url': user,
            'hotel_url': hotel_url,
            'overall_rating': overall,
            'service': random.randint(1, 5),
            'cleanliness': random.randint(1, 5),
            'value': random.randint(1, 5),
            'location': random.randint(1, 5),
            'rooms': random.randint(1, 5),
            'text': f'Sample review {r} for hotel {h}. Rating: {overall}/5.',
            'date': f'2020-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
        }
        reviews.append(review)
    total_reviews += len(reviews)

    fpath = os.path.join(raw_dir, f'{hotel_url}.json')
    with open(fpath, 'w') as f:
        json.dump(reviews, f)

print(f'  Generated {num_hotels} hotel files, {total_reviews} total reviews')
print(f'  Saved to {raw_dir}/')
" || {
    echo "ERROR: Sample generation failed."
    exit 1
}
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
        echo "  full   — download entire HotelRec dataset (~10GB)"
        echo "  sample — generate small synthetic dataset for testing"
        exit 1
        ;;
esac

# ─── Stats ────────────────────────────────────────────────────────────
echo ""
echo "--- Dataset stats ---"
N_FILES=$(ls "$RAW_DIR"/*.json 2>/dev/null | wc -l)
if [ "$N_FILES" -gt 0 ]; then
    TOTAL_SIZE=$(du -sh "$RAW_DIR" 2>/dev/null | cut -f1)
    echo "  JSON files: ${N_FILES}"
    echo "  Total size: ${TOTAL_SIZE}"

    # count total reviews across all files
    TOTAL_REVIEWS=$(python3 -c "
import json, glob
total = 0
for f in sorted(glob.glob('${RAW_DIR}/*.json')):
    try:
        with open(f) as fh:
            total += len(json.load(fh))
    except:
        pass
print(total)
" 2>/dev/null || echo "?")
    echo "  Total reviews: ${TOTAL_REVIEWS}"
else
    echo "  No JSON files found in ${RAW_DIR}/"
    echo "  Check the download instructions above."
fi
echo "---------------------"
