from pathlib import Path


def test_training_cli_uses_single_graph_method():
    text = Path("scripts/train.py").read_text(encoding="utf-8")
    assert "--city" in text
    assert "CrGeoTrajectoryPredictionModel" in text


def test_paper_graph_relations_are_present():
    text = Path("model/gnn_dataset.py").read_text(encoding="utf-8")
    for relation in ("v2v", "v2l", "l2v", "l2l", "vtv"):
        assert f'"{relation}"' in text
