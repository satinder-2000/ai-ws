import numpy as np
import pandas as pd
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

SEED = 42
np.random.seed(SEED)

from sklearn.preprocessing import StandardScaler, LabelBinarizer

class FeatureBinarizerAndScaler:
    """ This class needed for scales and factorize features
    """
    NUMERICAL_FEATURES = list()
    CATEGORICAL_FEATURES = list()
    BIN_FEATURES = list()
    binarizer = dict()
    scalers = dict()

    def __init__(self, numerical=list, categorical=list, binfeatures= list(), binarizer=dict(), scalers=dict()):
        self.NUMERICAL_FEATURES = numerical
        self.CATEGORICAL_FEATURES = categorical
        self.BIN_FEATURES = binfeatures
        self.BINARIZER = binarizer
        self.SCALERS = scalers

    def fit(self, trian_set):
        for feature in trian_set.columns:
            if feature.split('_')[-1] == 'cat':
                self.CATEGORICAL_FEATURES.append(feature)
            elif feature.split('_')[-1] != 'bin':
                self.NUMERICAL_FEATURES.append(feature)

        for feature in self.NUMERICAL_FEATURES:
            scaler = StandardScaler()
            self.scalers[feature] = scaler.fit(
                np.float64(trian_set[feature]).reshape((len(trian_set[feature]), 1))
            )

        for feature in self.CATEGORICAL_FEATURES:
            binarizer = LabelBinarizer()
            self.binarizer[feature] = binarizer.fit(trian_set[feature])

    def transform(self, data):
        binarizedAndScaledFeatures = np.empty((0,0))
        for feature in self.NUMERICAL_FEATURES:
            if feature in self.CATEGORICAL_FEATURES[0]:
                binarizedAndScaledFeatures = self.scalers[feature].transform(np.float64(data[feature]).reshape((len(data[feature]), 1)))
            else:
                binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures, self.scalers[feature].tranform(np.float64(data[feature]).reshape(len(data[feature])))), axis=1)

        for feature in self.CATEGORICAL_FEATURES:
            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures,
                                                         self.binarizers[feature].transform(data[feature])), axis=1)

            for feature in self.BIN_FEATURES:
                binarizedAndScaledFeatures= np.concatenate((binarizedAndScaledFeatures, np.array(data[feature]).reshape((len(data[feature]),
                                                                                       1))), axis=1)
                print(binarizedAndScaledFeatures.shape)

        return binarizedAndScaledFeatures
    