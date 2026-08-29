#!/usr/bin/env python3
"""Fast scan of CommonRoad files and location-disjoint train/val/test split.

This scanner deliberately reads only benchmark IDs and vehicle time steps. It
does not build lane geometry, so splitting hundreds of scenarios is fast.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.config import SEED, OBS_STEPS, PRED_STEPS, WINDOW_STRIDE, MIN_CONTEXT_VEHICLES, MAX_TARGETS_PER_WINDOW


def read_root(path: Path) -> ET.Element:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.endswith('.cr.xml')]
            if not members:
                members = [n for n in archive.namelist() if n.endswith('.xml')]
            if not members:
                raise ValueError('no XML member found')
            return ET.fromstring(archive.read(members[0]))
    return ET.parse(path).getroot()


def exact_time(state: ET.Element) -> int:
    node = state.find('time/exact')
    return int(round(float(node.text))) if node is not None and node.text else 0


def scan(path: Path) -> dict:
    root = read_root(path)
    benchmark = root.attrib.get('benchmarkID', path.stem)
    vehicles: dict[int, set[int]] = {}
    for obstacle in root.findall('dynamicObstacle'):
        times: set[int] = set()
        initial = obstacle.find('initialState')
        if initial is not None:
            times.add(exact_time(initial))
        for state in obstacle.findall('trajectory/state'):
            times.add(exact_time(state))
        if times:
            vehicles[int(obstacle.attrib['id'])] = times

    total_steps = OBS_STEPS + PRED_STEPS
    all_times = sorted(set().union(*vehicles.values())) if vehicles else []
    windows = samples = 0
    for start in range(0, max(0, len(all_times) - total_steps + 1), WINDOW_STRIDE):
        w = all_times[start:start + total_steps]
        if len(w) != total_steps or any(b != a + 1 for a, b in zip(w, w[1:])):
            continue
        obs = w[:OBS_STEPS]
        context = [vid for vid, ts in vehicles.items() if all(t in ts for t in obs)]
        targets = [vid for vid in context if all(t in vehicles[vid] for t in w)]
        if len(context) >= MIN_CONTEXT_VEHICLES and targets:
            windows += 1
            samples += min(len(targets), MAX_TARGETS_PER_WINDOW)
    return {
        'file': path.name,
        'benchmark_id': benchmark,
        'location_group': benchmark.split('-')[0],
        'windows': windows,
        'estimated_samples': samples,
    }


def assign_locations(rows: list[dict], seed: int) -> dict[str, str]:
    groups = defaultdict(list)
    for row in rows:
        groups[row['location_group']].append(row)
    items = [(loc, sum(int(r['estimated_samples']) for r in rr)) for loc, rr in groups.items()]
    rng = random.Random(seed)
    rng.shuffle(items)
    items.sort(key=lambda x: x[1], reverse=True)
    total = sum(n for _, n in items)
    targets = {'train': .70 * total, 'val': .15 * total, 'test': .15 * total}
    current = {k: 0 for k in targets}
    result = {}
    for loc, count in items:
        split = min(targets, key=lambda k: current[k] / max(targets[k], 1.0))
        result[loc] = split
        current[split] += count
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--output-root', default=str(ROOT / 'data'))
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--copy-files', action='store_true')
    args = p.parse_args()
    src, out = Path(args.input_dir), Path(args.output_root)
    (out/'splits').mkdir(parents=True, exist_ok=True)
    for split in ('train','val','test'):
        d=out/'raw'/split; d.mkdir(parents=True, exist_ok=True)
        if args.copy_files:
            for old in d.glob('*.xml'): old.unlink()

    eligible=[]; excluded=[]; files=sorted(src.glob('*.xml'))
    for i,path in enumerate(files,1):
        try:
            row=scan(path)
            if row['estimated_samples']>0: eligible.append(row)
            else:
                row['reason']='no valid 8-observation/12-prediction window'; excluded.append(row)
        except Exception as exc:
            excluded.append({'file':path.name,'benchmark_id':path.stem,'location_group':path.stem.split('-')[0],'windows':0,'estimated_samples':0,'reason':str(exc)})
        if i % 25 == 0 or i == len(files):
            print(f'scanned {i}/{len(files)}', flush=True)

    loc_split=assign_locations(eligible,args.seed)
    for row in eligible:
        row['split']=loc_split[row['location_group']]
        if args.copy_files:
            shutil.copy2(src/row['file'], out/'raw'/row['split']/row['file'])

    fields=['file','benchmark_id','location_group','split','windows','estimated_samples']
    with (out/'splits'/'split_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(eligible)
    fields2=['file','benchmark_id','location_group','windows','estimated_samples','reason']
    with (out/'splits'/'excluded.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields2); w.writeheader(); w.writerows(excluded)

    summary={'source_files':len(files),'eligible_scenarios':len(eligible),'excluded_scenarios':len(excluded),'split_method':'location-disjoint 70/15/15 balanced by estimated VOI samples','seed':args.seed,'config':{'obs_steps':OBS_STEPS,'pred_steps':PRED_STEPS,'stride':WINDOW_STRIDE,'max_targets_per_window':MAX_TARGETS_PER_WINDOW},'splits':{}}
    for split in ('train','val','test'):
        rows=[r for r in eligible if r['split']==split]
        summary['splits'][split]={'scenarios':len(rows),'locations':sorted({r['location_group'] for r in rows}),'location_count':len({r['location_group'] for r in rows}),'windows':sum(int(r['windows']) for r in rows),'estimated_samples':sum(int(r['estimated_samples']) for r in rows)}
    (out/'splits'/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
