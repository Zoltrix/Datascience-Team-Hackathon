import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_binary_ads(df):
    ingredients = list(set([item for l in df['ingredients'].values.tolist() for item in l]))

    binary_ads = np.ndarray((len(df), len(ingredients)))

    id_columns = df[['cuisine', 'id']].values

    def fill_ingredients(row):
        for ingredient in row.ingredients:
            binary_ads[row.name, ingredients.index(ingredient)] = 1

    # TODO potential for parallel processing
    df.apply(fill_ingredients, axis=1)

    return pd.DataFrame(data=np.column_stack((id_columns, binary_ads)),
                        index=df.index, columns=['cuisine', 'id'] + ingredients)

def train_val_split(df, seed, train_prop):
    train, test = train_test_split(df, test_size=1-train_prop, random_state=seed)
    ytrain = train.pop('cuisine')
    xtrain = train.values
    ytest = test.pop('cuisine')
    xtest = test.values

    return xtrain, ytrain, xtest, ytest


def prepare(df, seed=123131, train_prop=0.8):
    binary_ads = prepare_binary_ads(df)
    return train_val_split(binary_ads, seed, train_prop)

if __name__ == "__main__":
    df = pd.read_json("../data/train.json")
    #out = explode(df[:5], ['id', 'cuisine'], 'ingredients')
    out = prepare_binary_ads(df)
    print(out.head())
    print(type(out))
