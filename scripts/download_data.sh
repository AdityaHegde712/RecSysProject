#!/bin/bash
# Download HotelRec dataset
#
# The dataset is hosted on SWITCHdrive (Swiss academic cloud): https://drive.switch.ch/index.php/s/n48smsdufhRA7fR
#
# It's a single large JSON file (~10GB) containing all 50M reviews. 
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

# Full download from SWITCHdrive
download_full() {
    echo ">>> Downloading HotelRec dataset from SWITCHdrive..."
    echo "    Source: https://drive.switch.ch/index.php/s/n48smsdufhRA7fR"
    echo "    The archive is ~10GB (zip containing ~365K hotel JSON files)."
    echo "    No extraction needed - our pipeline streams directly from the zip."
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
    echo "  (This will take a while - ~10GB file)"
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

    # No extraction needed - preprocess_zip.py streams directly from the zip.
    echo ""
    echo "  Download complete. No extraction needed."
    echo "  The preprocessing step reads directly from the zip file."
    echo ""
    echo "  Next:"
    echo "    python -m src.data.preprocess --kcore 20    # 20-core filter + parquet"
    echo "    python -m src.data.split --kcore 20         # train/val/test split"
    echo "  (HPC layer at extras/hpc/run_hpc.sh available as an alternative.)"
}

# Dispatch 
case "$MODE" in
    full)
        download_full
        ;;
    *)
        echo "Usage: bash scripts/download_data.sh full"
        echo "  Downloads HotelRec zip from SWITCHdrive (~50GB)."
        echo "  No extraction needed - preprocess_zip.py streams from the zip."
        echo ""
        echo "  For a quick smoke test with a subset, run preprocess.py with --max-reviews 500000."
        exit 1
        ;;
esac

# Stats
echo ""
echo "--- Dataset stats ---"
ZIP_FILE=$(ls "${RAW_DIR}"/*.zip 2>/dev/null | head -1)
if [ -n "$ZIP_FILE" ]; then
    SIZE=$(du -h "$ZIP_FILE" | cut -f1)
    echo "  Archive: $(basename "$ZIP_FILE") ($SIZE)"
    echo "  preprocess_zip.py will stream directly from this file"
else
    echo "  No zip file found in ${RAW_DIR}/"
fi
echo "---------------------"
