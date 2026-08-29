#!/usr/bin/env python3
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'data/splits/split_manifest.csv').open()))
loc={s:{r['location_group'] for r in rows if r['split']==s} for s in ('train','val','test')}
assert not (loc['train'] & loc['val'])
assert not (loc['train'] & loc['test'])
assert not (loc['val'] & loc['test'])
for s in ('train','val','test'):
    expected=sum(r['split']==s for r in rows)
    actual=len(list((ROOT/'data/raw'/s).glob('*.xml')))
    assert expected==actual,(s,expected,actual)
    print(f'{s}: {actual} scenarios, {len(loc[s])} locations')
print('OK: location-disjoint split and file counts verified')
