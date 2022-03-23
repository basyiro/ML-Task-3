from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Local imports
import settings


def prepare_data(dataframe):
    train, test = train_test_split(
        dataframe, test_size=0.3, random_state=42, shuffle=True
    )

    X_train = StandardScaler().fit_transform(train.drop(columns="Survive"))
    y_train = train["Survive"].values

    X_test = StandardScaler().fit_transform(test.drop(columns="Survive"))
    y_test = test["Survive"].values

    return X_train, X_test, y_train, y_test


def define_models():
    model_list = []

    # Neural Net Classifier
    mlp_clf = MLPClassifier(
        solver="adam",
        batch_size=1000,
        alpha=1e-5,
        activation="tanh",
        max_iter=500,
        early_stopping=False,
        verbose=True,
        hidden_layer_sizes=(50, 50),
        random_state=100,
    )
    model_list.append(mlp_clf)

    # Random Forest
    rf_clf = RandomForestClassifier(
        criterion="gini",
        n_estimators=1750,
        max_depth=7,
        min_samples_split=6,
        min_samples_leaf=6,
        max_features="auto",
        oob_score=True,
        random_state=13,
        n_jobs=-1,
        verbose=1,
    )
    model_list.append(rf_clf)

    # SVM Classifier
    svc_clf = SVC(gamma="auto")
    model_list.append(svc_clf)

    return model_list


def run_models(dataframe):
    results = defaultdict(dict)
    X_train, X_test, y_train, y_test = prepare_data(dataframe)
    models = define_models()

    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        model_name = type(model).__name__
        results[model_name]["Accuracy"] = f"{accuracy_score(y_test, y_pred)*100:.2f}%"
        results[model_name]["F1 Score"] = f"{f1_score(y_test, y_pred):.2f}"

        print(settings.G + f"{model_name} completed training" + settings.W)

    model_list = []

    for k, v in results.items():
        print(f"{k}")
        model_list.append(k)
        for k, v in v.items():
            print(f"{k} - {v}")
        print("\n")

    accuracy_list = sorted(
        model_list,
        key=lambda x: (results[x]["Accuracy"]),
        reverse=True,
    )

    f1_list = sorted(
        model_list,
        key=lambda x: (results[x]["F1 Score"]),
        reverse=True,
    )

    print(f"From highest accuracy to lowest: {accuracy_list}")
    print(f"From highest F1 score to lowest: {f1_list}")
