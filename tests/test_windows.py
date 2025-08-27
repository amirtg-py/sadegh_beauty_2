import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from src.tracklet_gnn import create_windows

def test_create_windows_basic():
    df = pd.DataFrame({
        'Time': [0, 1, 2, 3, 4, 5],
        'Azimuth': [10, 20, 30, 40, 50, 60]
    })
    wins = create_windows(df, window=2, step=1)
    assert len(wins) == 5
    for w in wins:
        assert not w.empty
        assert w['Time'].max() - w['Time'].min() < 2.001
