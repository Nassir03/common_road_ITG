# Dataset Setup

```python
CITY = "boston"
```

```text
Boston:
/kaggle/input/datasets/abdullge26z811/boston/boston_t0.2_cleaneddata
/kaggle/input/datasets/abdullge26z811/boston

Pittsburgh:
/kaggle/input/datasets/abdullge26z811/pittsburgh/pittsburgh_t0.2_cleaneddata
/kaggle/input/datasets/abdullge26z811/pittsburgh

Singapore:
/kaggle/input/datasets/abdullge26z811/singapore/singapore_t0.2_cleaneddata
/kaggle/input/datasets/abdullge26z811/singapore
```

Inspect:

```bash
python scripts/inspect_kaggle_data.py --city boston
```

Prepare:

```bash
python scripts/prepare_city.py --city boston --workers 4 --overwrite
```

The output is:

```text
data/boston/
  split_manifest.csv
  summary.json
  processed/
    train/
      *.scenario.pt
      sample_index.pt
    val/
      *.scenario.pt
      sample_index.pt
    test/
      *.scenario.pt
      sample_index.pt
```

