#!/usr/bin/env python
from pathlib import Path
for city in ["boston","pittsburgh","singapore"]:
    root=Path(f"/kaggle/input/datasets/abdullge26z811/{city}")
    files=list(root.rglob("*.xml")) if root.exists() else []
    print(f"{city:10s} exists={root.exists()} xml_files={len(files)} root={root}")
    if files: print("  first:",files[0])
