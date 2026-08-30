#!/usr/bin/env python
import argparse,json
from pathlib import Path

def load(p): return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--paper",required=True); ap.add_argument("--itg",required=True); args=ap.parse_args(); a,b=load(args.paper),load(args.itg)
    if a.get("city")!=b.get("city"): raise ValueError("Compare results from the same city")
    da=b["test_ADE_m"]-a["test_ADE_m"]; df=b["test_FDE_m"]-a["test_FDE_m"]
    print(f"City: {a['city']}")
    print(f"Paper V2V baseline: ADE={a['test_ADE_m']:.6f} m FDE={a['test_FDE_m']:.6f} m")
    print(f"ITG V2V:            ADE={b['test_ADE_m']:.6f} m FDE={b['test_FDE_m']:.6f} m")
    print(f"ITG - baseline:     dADE={da:+.6f} m dFDE={df:+.6f} m")
    if a["test_ADE_m"]: print(f"ADE change: {100*da/a['test_ADE_m']:+.2f}%")
    if a["test_FDE_m"]: print(f"FDE change: {100*df/a['test_FDE_m']:+.2f}%")
if __name__=="__main__": main()
