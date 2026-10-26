#!/usr/bin/env python3
"""Verify the two simulation datasets belong to the same trained model/controller."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'realtime_model'))
from run_provenance import verify_results

if __name__ == '__main__':
    result = verify_results(ROOT)
    print('CONSISTENT:', result)
