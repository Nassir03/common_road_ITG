#!/usr/bin/env python3
"""Compare radius-baseline and ITG result JSON files."""
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--radius", required=True); p.add_argument("--itg", required=True); a=p.parse_args()
r=json.loads(Path(a.radius).read_text()); i=json.loads(Path(a.itg).read_text())
print(f"Radius: ADE={r['test_ADE_m']:.4f} FDE={r['test_FDE_m']:.4f}")
print(f"ITG:    ADE={i['test_ADE_m']:.4f} FDE={i['test_FDE_m']:.4f}")
print(f"ADE change: {i['test_ADE_m']-r['test_ADE_m']:+.4f} m ({100*(i['test_ADE_m']/r['test_ADE_m']-1):+.2f}%)")
print(f"FDE change: {i['test_FDE_m']-r['test_FDE_m']:+.4f} m ({100*(i['test_FDE_m']/r['test_FDE_m']-1):+.2f}%)")
