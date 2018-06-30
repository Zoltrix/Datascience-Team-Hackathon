import pandas as pd


def explode(df, index_cols, list_col):
    return df\
        .set_index(index_cols) \
        .apply(lambda row: pd.Series(row[list_col]), axis=1) \
        .stack() \
        .reset_index(level=2, drop=True) \
        .to_frame(list_col) \
        .reset_index()


def prepare_binary_ads(df, keys, value):
    df = explode(df, keys, value)
    return pd.get_dummies(df, columns=['ingredient'], prefix='has')\
        .groupby(['id', 'cuisine'])\
        .sum(axis=1)\
        .reset_index()


if __name__ == "__main__":
    df = pd.read_json("../data/train.json")
    #out = explode(df[:5], ['id', 'cuisine'], 'ingredients')
    out = prepare_binary_ads(df[:5], ['cuisine', 'id'], 'ingredients')
    print(out.head())
    print(type(out))
    print(out.shape)
