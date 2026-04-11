import json
import csv
import argparse
from pathlib import Path


def jsonl_to_csv(input_path: str, output_path: str = None):
    input_file = Path(input_path)
    if output_path is None:
        output_path = input_file.with_suffix(".csv")

    records = []
    fieldnames = set()

    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {i}: {e}")
                continue

            # Flatten property_dict into top-level columns if present
            prop = record.pop("property_dict", {}) or {}
            for k, v in prop.items():
                record[f"prop_{k}"] = v

            records.append(record)
            fieldnames.update(record.keys())

    if not records:
        print("No valid records found.")
        return

    # Stable column order: known fields first, then any extras
    known = ["hotel_url", "author", "date", "rating", "title", "text"]
    extras = sorted(k for k in fieldnames if k not in known)
    fieldnames = known + extras

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            # Fill missing keys with empty string
            row = {k: record.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"Done: {len(records)} records written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL hotel reviews to CSV")
    parser.add_argument("input", help="Path to input .txt / .jsonl file")
    parser.add_argument("-o", "--output", help="Output CSV path (default: same name as input)")
    args = parser.parse_args()
    jsonl_to_csv(args.input, args.output)