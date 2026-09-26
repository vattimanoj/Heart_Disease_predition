import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from log import setup_logging
logger=setup_logging('feature_selection')
def best_col(X_train, X_test, y_train, y_test):
    try:
        logger.info(f"before constant technique :{X_train.columns},{X_train.shape},{X_test.columns},{X_test.shape}")
        #constant technique
        cons_obj=VarianceThreshold(threshold=0.0)
        cons_obj.fit(X_train)
        logger.info(f"columns to remove:{X_train.columns[~cons_obj.get_support()]}")
        X_train=X_train.drop(['fbs_yeo_trim'],axis=1)
        X_test=X_test.drop(['fbs_yeo_trim'],axis=1)
        logger.info(f"after constant technique:{X_train.columns},{X_train.shape},{X_test.columns},{X_test.shape}")
        #quasi constant
        quasi_obj=VarianceThreshold(threshold=0.1)
        quasi_obj.fit(X_train)
        logger.info(f"columns to remove:{X_train.columns[~quasi_obj.get_support()]}")
        X_train=X_train.drop(['trestbps_yeo_trim', 'chol_yeo_trim', 'exang_yeo_trim', 'ca_yeo_trim'],axis=1)
        X_test=X_test.drop(['trestbps_yeo_trim', 'chol_yeo_trim', 'exang_yeo_trim', 'ca_yeo_trim'],axis=1)
        logger.info(f'after quasi constant:{ X_train.columns},{X_train.shape},{X_test.columns},{X_test.shape}')
        #corration with hypothesis testing
        # corr_p_values = []
        # for i in X_train.columns:
        #     values = pearsonr(X_train[i] , y_train)
        #     corr_p_values.append(values)
        # corr_p_values = np.array(corr_p_values)
        # p_values = corr_p_values[:, 1]
        # plt.figure(figsize=(5, 3))
        # plt.title("Hypothesis Testing")
        # plt.xlabel("column Names")
        # plt.ylabel("P_values for Each Independent column")
        # plt.bar(X_train.columns, p_values)
        # plt.show()
        X_train = X_train.drop(['restecg_yeo_trim'], axis=1)
        X_test = X_test.drop(['restecg_yeo_trim'], axis=1)
        logger.info(f"after hypothesis testing :{X_train.columns},{X_train.shape},{X_test.columns},{X_test.shape}")
        return X_train, X_test


    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")