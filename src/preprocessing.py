import pandas as pd

# Local imports
import settings

# Scan the dataframe to find any missing data
def find_null_features(dataframe):

    null_features = [
        feature
        for feature in dataframe.columns
        if dataframe[feature].isnull().sum() > 1
    ]

    for feature in null_features:
        print(
            settings.O
            + f"{feature} has {dataframe[feature].isnull().sum()} missing values"
            + settings.W
        )

    return null_features


# Scan the dataframe to find any numerical data that is less than 0
def find_negative_values(dataframe):

    negative_features = []

    for numerical_feature in settings.numerical_feature_list:
        check_less_than_zero_df = dataframe.loc[dataframe[numerical_feature] < 0]
        if len(check_less_than_zero_df) > 0:
            negative_features.append(numerical_feature)
            print(
                settings.O
                + f"{numerical_feature} has {len(check_less_than_zero_df)} negative values"
                + settings.W
            )

    return negative_features


# Fix null_features, negative_features and also features with repeated labellings
def fix_features(dataframe, null_features, negative_features):

    for null_feature in null_features:
        # Since we found from EDA all the missing numbers were from Creatinine and non-survivors
        # we do a special condition for Creatinine.
        if null_feature == "Creatinine":
            survived_df = dataframe[dataframe["Survive"] == 1]
            not_survive_df = dataframe[dataframe["Survive"] == 0]
            not_survive_df["Creatinine"] = not_survive_df.groupby(["Gender"])[
                "Creatinine"
            ].apply(lambda value: value.fillna(value.median()))
            dataframe = pd.concat([survived_df, not_survive_df])
        else:
            # If we somehow have other null_features, we just use the median of the entire df
            # while still grouping by gender
            dataframe[null_feature] = dataframe.groupby(["Gender"])[null_feature].apply(
                lambda value: value.fillna(value.median())
            )

    # Convert the negative values to positive by taking the absolute
    for negative_feature in negative_features:
        dataframe[negative_feature] = dataframe[negative_feature].abs()

    # From EDA, we found that "Smoke" and "Ejection Fraction" have repeated labellings
    dataframe = dataframe.replace({"Smoke": {"NO": "No", "YES": "Yes"}})
    dataframe = dataframe.replace({"Ejection Fraction": {"L": "Low", "N": "Normal"}})

    return dataframe


def preprocessing(dataframe):
    # From EDA, "ID" has duplicates
    dataframe = dataframe.drop_duplicates(subset=["ID"], keep=False)
    # From EDA, "Survive" label is inconsistent
    dataframe = dataframe.replace({"Survive": {"No": 0, "Yes": 1, "0": 0, "1": 1}})

    null_features = find_null_features(dataframe)
    negative_features = find_negative_values(dataframe)
    print(settings.G + "Fixing errorneous values" + settings.W)
    dataframe = fix_features(dataframe, null_features, negative_features)

    return dataframe
