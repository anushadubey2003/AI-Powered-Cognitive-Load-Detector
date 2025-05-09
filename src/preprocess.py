import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.dropna()
    le = LabelEncoder()
    df["LoadLabel"] = le.fit_transform(df["LoadLevel"])
    return df, le
