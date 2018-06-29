import pandas as pd


def explode(df, index_cols, list_col):
    rows = []
    for i, row in df.iterrows():
        for a in row[list_col]:
            new_row = row[index_cols].values.tolist()
            new_row.extend([a])
            rows.append(new_row)

    return pd.DataFrame(rows, columns=df.columns)


def prepare_binary_ads(df, keys, value):
    df = explode(df, keys, value)
    return df.pivot_table(index=keys, columns=value, aggfunc=[len], fill_value=0)


if __name__ == "__main__":
    df = pd.read_json("../data/train.json")
    #out = explode(df[:5], ['id', 'cuisine'], 'ingredients')
    out = prepare_binary_ads(df[:5], ['cuisine', 'id'], 'ingredients')
    print(out.head())
    print(type(out))
    print(out.shape)
