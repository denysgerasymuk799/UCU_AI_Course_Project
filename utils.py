import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


def scale_normalize(df, features):
    # separate the features
    feature_data = df.loc[:, features].values

    # scale and center data (mean = 0, variance = 1)
    scaled_data = StandardScaler().fit_transform(feature_data)

    # show the result
    scaled_df = pd.DataFrame(data=scaled_data, columns=features)
    return scaled_df
