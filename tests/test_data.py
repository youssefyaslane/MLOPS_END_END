from pathlib import Path
from src.data_preprocessing import load_data, make_preprocessor
from src.utils import DATA_PATH

def test_preprocessing_smoke():
    assert Path(DATA_PATH).exists(), "Dataset missing in data/telco_churn.csv"
    df = load_data(str(DATA_PATH))
    X, y, pre = make_preprocessor(df)
    assert len(X) == len(y) and len(X) > 0
