"""Measure existing lossless Zstd binding on an LNT without changing the source."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gui import native_codecs as codecs
CODEC_ZSTD = codecs.CODEC_ZSTD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    raw_bytes = stored_bytes = 0
    encode_seconds = decode_seconds = 0.0
    original_hash, restored_hash = hashlib.sha256(), hashlib.sha256()
    with args.dataset.open('rb') as source:
        while raw := source.read(16 << 20):
            started = time.perf_counter()
            stored = codecs.compress(raw, CODEC_ZSTD, level=1)
            encode_seconds += time.perf_counter()-started
            started = time.perf_counter()
            restored = codecs.decompress(stored, len(raw), CODEC_ZSTD)
            decode_seconds += time.perf_counter()-started
            assert restored == raw, 'lossless round-trip failed'
            original_hash.update(raw)
            restored_hash.update(restored)
            raw_bytes += len(raw)
            stored_bytes += len(stored)
    assert original_hash.digest() == restored_hash.digest()
    print(json.dumps({'versions': codecs.versions(), 'raw_mb': raw_bytes/1e6,
                      'compressed_mb': stored_bytes/1e6, 'ratio': raw_bytes/stored_bytes,
                      'compression_seconds': encode_seconds, 'decompression_seconds': decode_seconds,
                      'decompression_mb_per_second': raw_bytes/1e6/decode_seconds,
                      'byte_identical': True, 'sha256': original_hash.hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
