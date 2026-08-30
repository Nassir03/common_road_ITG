from scripts.prepare_city import KAGGLE_CANDIDATES

def test_requested_kaggle_paths_are_present():
    assert str(KAGGLE_CANDIDATES["boston"][0]) == "/kaggle/input/datasets/abdullge26z811/boston/boston_t0.2_cleaneddata"
    assert str(KAGGLE_CANDIDATES["pittsburgh"][0]) == "/kaggle/input/datasets/abdullge26z811/pittsburgh/pittsburgh_t0.2_cleaneddata"
    assert str(KAGGLE_CANDIDATES["singapore"][1]) == "/kaggle/input/datasets/abdullge26z811/singapore"
