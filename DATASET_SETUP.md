## Expected folder layout after CommonRoad conversion

For each city experiment, prepare its own root so training and validation remain within that city:

```text
Singapore/
  raw/train/*.xml
  raw/val/*.xml
  raw/test/*.xml       # optional if you maintain a held-out set

Boston/
  raw/train/*.xml
  raw/val/*.xml
  raw/test/*.xml

Pittsburgh/
  raw/train/*.xml
  raw/val/*.xml
  raw/test/*.xml
```

Then preprocess one city, for example:

```bash
python scripts/preprocess.py \
  --raw-root /path/to/Singapore/raw \
  --processed-root /path/to/Singapore/processed
```

Train only on that city's processed root:

```bash
python scripts/train.py \
  --data-root /path/to/Singapore/processed \
  --output outputs/singapore_model.pt
```

Repeat independently for Boston and Pittsburgh.
