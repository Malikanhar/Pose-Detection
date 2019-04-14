import pandas as pd

def load_dataset(filename):
    df = pd.read_excel(filename)
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    return X,y

def to_timeseries(X, pattern, series_length):
    if pattern == "xyxy":
        pass

if __name__ == '__main__':
    print("A")
    X,y = load_dataset("DATA_SPLIT.xlsx")
    print(X.shape)