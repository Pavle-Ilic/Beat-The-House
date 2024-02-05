import pandas as pd


def save(PATH, qTable):
    dataFrame = pd.DataFrame(qTable)
    dataFrame.to_csv(PATH, index=False)


def load(PATH):
    qTable_Loaded = pd.read_csv(PATH)
    qTable = qTable_Loaded.to_numpy()