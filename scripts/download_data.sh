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
    echo "    This is ~10GB. Make sure you have enough disk space."
    echo ""

    DEST="${RAW_DIR}/HotelRec.json"
    DOWNLOAD_URL="https://drive.switch.ch/index.php/s/n48smsdufhRA7fR/download"

    if [ -f "$DEST" ]; then
        SIZE=$(du -h "$DEST" | cut -f1)
        echo "  File already exists: $DEST ($SIZE)"
        echo "  Delete it first if you want to re-download."
        echo ""
    else
        echo "  Downloading to $DEST ..."
        echo "  (This will take a while — ~10GB file)"
        echo ""

        # Try wget first (better for large files, supports resume)
        if command -v wget &> /dev/null; then
            wget -c -O "$DEST" "$DOWNLOAD_URL" || {
                echo ""
                echo "ERROR: wget failed. Try manually:"
                echo "  wget -O $DEST '$DOWNLOAD_URL'"
                exit 1
            }
        elif command -v curl &> /dev/null; then
            curl -L -C - -o "$DEST" "$DOWNLOAD_URL" || {
                echo ""
                echo "ERROR: curl failed. Try manually:"
                echo "  curl -L -o $DEST '$DOWNLOAD_URL'"
                exit 1
            }
        else
            echo "ERROR: Neither wget nor curl found."
            echo "Download manually from: https://drive.switch.ch/index.php/s/n48smsdufhRA7fR"
            echo "Save to: $DEST"
            exit 1
        fi
    fi

    # If it's a zip/tar, extract it
    if file "$DEST" 2>/dev/null | grep -q "Zip archive"; then
        echo "  Extracting zip archive..."
        unzip -o "$DEST" -d "$RAW_DIR/"
    elif file "$DEST" 2>/dev/null | grep -q "gzip"; then
        echo "  Extracting gzip archive..."
        gunzip -k "$DEST" 2>/dev/null || gzip -dk "$DEST"
    fi

    echo ""
    echo "  Download complete."
}

# ─── Sample: generate synthetic data ─────────────────────────────────
download_sample() {
    echo ">>> Creating sample dataset for development/testing..."
    echo "    Generating synthetic hotel review data matching HotelRec format."
    echo ""

    python3 -c "
import json
import random
import os

random.seed(42)
raw_dir = '${RAW_DIR}'

# HotelRec actual format (from the repo README):
# {
#   'hotel_url': '...',
#   'author': 'username',
#   'date': '2010-02-01T00:00:00',
#   'rating': 4.0,
#   'title': 'Great customer service',
#   'text': '...',
#   'property_dict': {'sleep quality': 4.0, 'value': 4.0, ...}
# }

sub_rating_keys = ['sleep quality', 'value', 'rooms', 'service', 'cleanliness', 'location']
user_pool = [f'user_{i}' for i in range(500)]
hotel_names = [
    'Hotel_Review-g{}-d{}-Reviews-Hotel_{}'.format(
        random.randint(100000, 999999),
        random.randint(100000, 9999999),
        f'TestHotel_{h}'
    ) for h in range(50)
]

all_reviews = []
for hotel_url in hotel_names:
    n_reviews = random.randint(30, 300)
    for _ in range(n_reviews):
        overall = float(random.randint(1, 5))
        # ~70% of reviews have sub-ratings (matching paper stats)
        prop = {}
        if random.random() < 0.71:
            for key in sub_rating_keys:
                if random.random() < 0.9:  # not all sub-ratings always present
                    prop[key] = float(random.randint(1, 5))

        review = {
            'hotel_url': hotel_url,
            'author': random.choice(user_pool),
            'date': f'{random.randint(2005, 2019)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}T00:00:00',
            'rating': overall,
            'title': f'Review title {random.randint(1, 10000)}',
            'text': ' '.join(['word'] * random.randint(20, 300)),
            'property_dict': prop,
        }
        all_reviews.append(review)

# Save as single JSON file (matching the actual HotelRec distribution format)
fpath = os.path.join(raw_dir, 'HotelRec.json')
with open(fpath, 'w') as f:
    json.dump(all_reviews, f)

print(f'  Generated {len(hotel_names)} hotels, {len(all_reviews)} reviews')
print(f'  Unique users: {len(set(r[\"author\"] for r in all_reviews))}')
print(f'  Saved to {fpath}')
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
        echo "  full   — download from SWITCHdrive (~10GB)"
        echo "  sample — generate synthetic data for testing"
        exit 1
        ;;
esac

# ─── Stats ────────────────────────────────────────────────────────────
echo ""
echo "--- Dataset stats ---"
if [ -f "${RAW_DIR}/HotelRec.json" ]; then
    SIZE=$(du -h "${RAW_DIR}/HotelRec.json" | cut -f1)
    echo "  File: HotelRec.json ($SIZE)"

    python3 -c "
import json
with open('${RAW_DIR}/HotelRec.json') as f:
    data = json.load(f)
users = set(r.get('author', r.get('user_url', '')) for r in data)
hotels = set(r['hotel_url'] for r in data)
print(f'  Reviews: {len(data):,}')
print(f'  Users:   {len(users):,}')
print(f'  Hotels:  {len(hotels):,}')
" 2>/dev/null || echo "  (Could not parse — file may still be downloading)"
else
    N_FILES=$(ls "$RAW_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "  JSON files: ${N_FILES}"
    if [ "$N_FILES" -gt 0 ]; then
        TOTAL_SIZE=$(du -sh "$RAW_DIR" 2>/dev/null | cut -f1)
        echo "  Total size: ${TOTAL_SIZE}"
    fi
fi
echo "---------------------"
