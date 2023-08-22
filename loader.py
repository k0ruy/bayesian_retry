import pandas as pd
from pathlib import Path


def load_data(path_to_forlder: str):
    df = pd.concat([pd.read_csv(Path(path_to_forlder, "hockey_15-16.csv"), sep=";"), 
                    pd.read_csv(Path(path_to_forlder, "hockey_16-17.csv"), sep=";"), 
                    pd.read_csv(Path(path_to_forlder, "hockey_17-18.csv"), sep=";"), 
                    pd.read_csv(Path(path_to_forlder, "hockey_18-19.csv"), sep=";"), 
                    pd.read_csv(Path(path_to_forlder, "hockey_19-20.csv"), sep=";"), 
                    pd.read_csv(Path(path_to_forlder, "hockey_20-21.csv"), sep=";"),
                    pd.read_csv(Path(path_to_forlder, "hockey_21-22.csv"), sep=";"),], axis=0, ignore_index=1)
    
    return df