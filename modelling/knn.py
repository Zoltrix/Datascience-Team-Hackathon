from sklearn.neighbors import KNeighborsClassifier


def knn(xtrain, ytrain, **kwargs):
    k_nn = KNeighborsClassifier(**kwargs)

    return k_nn.fit(xtrain, ytrain)


def eval(mdl, xtest, ytest):
    # probs = knn.predict_proba(xtest)

    return mdl.score(xtest, ytest)


def run(xtrain, ytrain, xtest, ytest, **kwargs):
    mdl = knn(xtrain, ytrain, **kwargs)

    return eval(mdl, xtest, ytest)

if __name__ == "__main__":
    from loading.load_data import load_raw_data
    from data_preparation.data_preparation import prepare
    df = load_raw_data("../data/train.json")
    xtrain, ytrain, xtest, ytest = prepare(df[:100], ['id', 'cuisine'], 'ingredients')

    print(run(xtrain, ytrain, xtest, ytest))