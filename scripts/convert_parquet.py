import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def jsonl_to_parquet(
    input_file: str,
    output_file: str,
    chunk_size: int = 50000,
    compression: str = 'snappy'
) -> None:
    """
    Convert a large JSONL file to Parquet with streaming/chunked processing.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output Parquet file
        chunk_size: Number of rows to process at once (default 50k)
        compression: Parquet compression codec ('snappy', 'gzip', 'brotli', 'none')
    """
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    logger.info(f"Starting conversion: {input_file} -> {output_file}")
    logger.info(f"Chunk size: {chunk_size:,} rows, Compression: {compression}")
    
    file_size_gb = input_path.stat().st_size / (1024**3)
    logger.info(f"Input file size: {file_size_gb:.2f} GB")
    
    writer = None
    chunk = []
    rows_processed = 0
    rows_failed = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    chunk.append(_normalize_record(record))
                    rows_processed += 1
                except json.JSONDecodeError as e:
                    rows_failed += 1
                    if rows_failed <= 5:
                        logger.warning(f"Skipping malformed JSON at line {line_num}: {str(e)[:100]}")
                    continue
                
                # Process chunk when it reaches desired size
                if len(chunk) >= chunk_size:
                    df = pd.DataFrame(chunk)
                    
                    # Infer and optimize dtypes on first chunk
                    if writer is None:
                        schema = _get_fixed_schema()
                        writer = pq.ParquetWriter(output_file, schema, compression=compression)
                        logger.info(f"Parquet schema initialized: {schema}")
                    
                    # Convert df with explicit cast before write:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    table = pa.Table.from_pandas(df, schema=_get_fixed_schema(), safe=False)
                    writer.write_table(table)
                    
                    logger.info(f"Processed {rows_processed:,} rows ({rows_processed / 1e6 * file_size_gb / (input_path.stat().st_size / (1024**3)) * 100:.1f}% est.)")
                    chunk = []
        
        # Write final chunk
        if chunk:
            df = pd.DataFrame(chunk)
            if writer is None:
                schema = _get_fixed_schema()
                writer = pq.ParquetWriter(output_file, schema, compression=compression)
            
            # Convert df with explicit cast before write:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            table = pa.Table.from_pandas(df, schema=_get_fixed_schema(), safe=False)
            writer.write_table(table)
        
        if writer:
            writer.close()
        
        output_size_gb = output_path.stat().st_size / (1024**3)
        compression_ratio = (1 - output_size_gb / file_size_gb) * 100
        
        logger.info("=" * 60)
        logger.info(f"Conversion completed successfully!")
        logger.info(f"Total rows processed: {rows_processed:,}")
        logger.info(f"Rows failed (malformed JSON): {rows_failed:,}")
        logger.info(f"Input size: {file_size_gb:.2f} GB")
        logger.info(f"Output size: {output_size_gb:.2f} GB")
        logger.info(f"Compression ratio: {compression_ratio:.1f}%")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        if writer:
            try:
                writer.close()
            except Exception as e:
                logger.error(f"Failed to close writer: {str(e)}")
        # Clean up incomplete file
        if output_path.exists():
            output_path.unlink()
        raise


# def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Optimize DataFrame dtypes for better compression and memory usage.
#     """
#     for col in df.columns:
#         col_type = df[col].dtype
        
#         # Handle datetime strings
#         if col == 'date' and col_type == 'object':
#             try:
#                 df[col] = pd.to_datetime(df[col])
#                 continue
#             except:
#                 pass
        
#         # Optimize numeric columns
#         if col_type in ['float64', 'int64']:
#             if col in ['rating']:
#                 df[col] = df[col].astype('float32')
#             elif col_type == 'int64':
#                 if df[col].max() < 2**31:
#                     df[col] = df[col].astype('int32')
        
#         # String columns stay as object (PyArrow handles efficiently)
    
#     return df


PROPERTY_KEYS = ['service', 'cleanliness', 'sleep_quality', 'location', 'value', 'rooms']

def _normalize_record(record: dict) -> dict:
    """Flatten property_dict, clean corrupt chars, enforce fixed schema."""
    prop = record.pop('property_dict', {}) or {}
    for key in PROPERTY_KEYS:
        # 'sleep quality' in data -> 'sleep_quality' column
        raw_key = key.replace('_', ' ')
        record[key] = float(prop[raw_key]) if raw_key in prop else None

    # Strip non-UTF8 / replacement chars from string fields
    for field in ('hotel_url', 'author', 'title', 'text'):
        val = record.get(field)
        if isinstance(val, str):
            record[field] = val.encode('utf-8', errors='ignore').decode('utf-8').replace('\ufffd', '')

    return record


def _get_fixed_schema() -> pa.Schema:
    return pa.schema([
        ('hotel_url',    pa.string()),
        ('author',       pa.string()),
        ('date',         pa.timestamp('ms')),
        ('rating',       pa.float32()),
        ('title',        pa.string()),
        ('text',         pa.string()),
        ('service',      pa.float32()),
        ('cleanliness',  pa.float32()),
        ('sleep_quality',pa.float32()),
        ('location',     pa.float32()),
        ('value',        pa.float32()),
        ('rooms',        pa.float32()),
    ])


def read_parquet_sample(parquet_file: str, n_rows: int = 1000) -> pd.DataFrame:
    """
    Read a sample of rows from the Parquet file for inspection.
    """
    return pd.read_parquet(parquet_file, engine='pyarrow').head(n_rows)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python jsonl_to_parquet.py <input_jsonl> <output_parquet> [chunk_size] [compression]")
        print("Example: python jsonl_to_parquet.py data.jsonl data.parquet 50000 snappy")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    chunk_size = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
    compression = sys.argv[4] if len(sys.argv) > 4 else 'snappy'
    
    jsonl_to_parquet(input_file, output_file, chunk_size, compression)