import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import yeojohnson

import sys
from log import setup_logging
logger = setup_logging("variable_transformation")
def outliers(X_train,X_test):
    try:
        logger.info(f"Before X_train column names : {X_train.columns}")
        logger.info(f"Before X_test column names : {X_test.columns}")
        for i in X_train.columns:
            X_train[i + "_yeo"], lam_value = yeojohnson(X_train[i])
            X_test[i + "_yeo"], lam_value = yeojohnson(X_test[i])
            X_train = X_train.drop([i], axis=1)
            X_test = X_test.drop([i], axis=1)
            iqr = X_train[i + "_yeo"].quantile(0.75) - X_train[i + "_yeo"].quantile(0.25)
            upper_limit = X_train[i + "_yeo"].quantile(0.75) + (1.5 * iqr)
            lower_limit = X_train[i + "_yeo"].quantile(0.25) - (1.5 * iqr)

            X_train[i + "_yeo_trim"] = np.where(X_train[i + "_yeo"] > upper_limit, upper_limit,np.where(X_train[i + "_yeo"] < lower_limit, lower_limit,X_train[i + "_yeo"]))
            X_test[i + "_yeo_trim"] = np.where(X_test[i + "_yeo"] > upper_limit, upper_limit,np.where(X_test[i + "_yeo"] < lower_limit, lower_limit,X_test[i + "_yeo"]))
            X_train = X_train.drop([i + "_yeo"], axis=1)
            X_test = X_test.drop([i + "_yeo"], axis=1)

        logger.info(f"After X_train column names : {X_train.columns}")
        logger.info(f"After X_test column names : {X_test.columns}")

        return X_train, X_test


        # for i in X_train.columns:
        #     plt.figure(figsize=(8, 3))
        #     plt.subplot(1, 2, 1)
        #     plt.title("Normal Distribution")
        #     X_train[i].plot(kind='kde', color='r')
        #
        #     plt.subplot(1, 2, 2)
        #     sns.boxplot(x=X_train[i])
        #
        #     plt.show()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")