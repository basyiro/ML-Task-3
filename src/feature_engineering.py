import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Local imports
import settings


def bin_features(dataframe):
    # From EDA, we want to bin these features
    binned_features = ["Sodium", "Creatinine", "Platelets", "Creatine phosphokinase"]
    for feature in binned_features:
        dataframe[feature] = pd.qcut(dataframe[feature], 5)
        dataframe[feature] = LabelEncoder().fit_transform(dataframe[feature])

    return dataframe


def combine_features(dataframe):
    # From EDA  we want to combine these features
    dataframe = dataframe.replace(
        {"Ejection Fraction": {"Normal": "Normal/High", "High": "Normal/High"}}
    )

    return dataframe


def encode_categorical(dataframe):

    encoded_features = []
    drop_columns = []

    for feature in settings.categorical_feature_list:
        try:
            encoded_feat = (
                OneHotEncoder()
                .fit_transform(dataframe[feature].values.reshape(-1, 1))
                .toarray()
            )
            n = dataframe[feature].nunique()
            cols = ["{}_{}".format(feature, n) for n in range(1, n + 1)]
            encoded_df = pd.DataFrame(encoded_feat, columns=cols)
            encoded_df.index = dataframe.index
            encoded_features.append(encoded_df)
            # Add to drop_columns after encoding the feature
            drop_columns.append(feature)
        except:
            print(settings.O + f'"{feature}" no longer in dataframe' + settings.W)

    dataframe = pd.concat([dataframe, *encoded_features], axis=1)
    dataframe = dataframe.drop(drop_columns, axis=1)

    return dataframe


def feature_engineering(dataframe):
    # From EDA, we drop "Favorite Color", "Height" and "ID"
    dataframe = dataframe.drop(["Favorite color", "Height", "ID"], axis=1)
    dataframe = bin_features(dataframe)
    dataframe = combine_features(dataframe)
    dataframe = encode_categorical(dataframe)
    return dataframe
