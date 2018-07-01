from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve, ShuffleSplit
import matplotlib.pyplot as plt
import cProfile

def xgboost(xtrain, ytrain, **kwargs):
    xgb = XGBClassifier(**kwargs)

    return xgb.fit(xtrain, ytrain)


def eval(mdl, xtest, ytest):
    # probs = knn.predict_proba(xtest)

    return mdl.score(xtest, ytest)


def run(xtrain, ytrain, xtest, ytest, **kwargs):
    print("Training xgboost ...")
    mdl = xgboost(xtrain, ytrain, **kwargs)
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    from modelling.learning_curve import plot_learning_curve

    print("Generating learning curve ...")
    plot_learning_curve(mdl, 'xgb_lr_curve', xtrain, ytrain, (0., 1.), cv=None, n_jobs=-1)
    plt.show()

    print("Evaluating xgboost ...")
    return eval(mdl, xtest, ytest)

if __name__ == "__main__":
    from loading.load_data import load_raw_data
    from data_preparation.data_preparation import prepare
    df = load_raw_data("../data/train.json")
    cProfile.run("prepare(df[:1000], ['id', 'cuisine'], 'ingredients')")
    #xtrain, ytrain, xtest, ytest = prepare(df, ['id', 'cuisine'], 'ingredients')

    #print(run(xtrain, ytrain, xtest, ytest, n_jobs=-1))
