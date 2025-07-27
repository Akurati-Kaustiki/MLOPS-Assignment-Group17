from sklearn.datasets import load_iris
import pandas as pd

def save_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])
    df.to_csv('data/iris.csv', index=False)
    print("Dataset saved to data/iris.csv")

if __name__ == "__main__":
    save_data()
