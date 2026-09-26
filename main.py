"""
Heart disease prediction pipeline.
    1. Load the data and split it into train/test sets.
    2. Transform variables and trim outliers (variable_transformation.outliers).
    3. Select the most useful features (feature_selection.best_col).
    4. Balance the classes with SMOTE and scale the features.
    5. Train and compare several candidate models (all_model.common).
    6. Train the final chosen model (GaussianNB), evaluate it, and
       save the trained model and scaler to disk as model.pkl / scaled.pkl.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings("ignore")
from log import setup_logging
logger = setup_logging('main')
from variable_transformation import outliers
from feature_selection import best_col
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from all_model import common
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle
import warnings
warnings.filterwarnings("ignore")
class HEART:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(path)
            logger.info(f"Null values in the data:\n{self.df.isnull().sum()}")
            self.x=self.df.iloc[:,:-1]
            self.y=self.df.iloc[:,-1]
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.2,random_state=42)
            logger.info(f"Training dataset size : \n {self.X_train.shape}  => {self.y_train.shape}")
            logger.info(f"Testing dataset size : \n {self.X_test.shape} => {self.y_test.shape}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def vt_outliers(self):
        try:
            self.X_train,self.X_test=outliers(self.X_train,self.X_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
    def feature_selection(self):
        try:
            self.X_train,self.X_test= best_col(self.X_train,self.X_test,self.y_train,self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def data_balancing(self):
        try:
            logger.info(f"Data balancing : \n {self.X_train.shape} => {self.y_train.shape}")
            logger.info(f"no of rows in target:{1}:{sum(self.y_train==1)}")
            logger.info(f"no of columns in target:{0}:{sum(self.y_train==0)}")
            sm_obj=SMOTE(random_state=42)
            self.X_train_bal,self.y_train_bal=sm_obj.fit_resample(self.X_train,self.y_train)
            logger.info(f"After Data balancing : \n {self.X_train_bal.shape} => {self.y_train_bal.shape}")
            logger.info(f"no of rows in target:{1}:{sum(self.y_train_bal == 1)}")
            logger.info(f"no of columns in target:{0}:{sum(self.y_train_bal== 0)}")
            #Scale down using Z score
            self.sc=StandardScaler()
            self.X_train_bal_scaled=self.sc.fit_transform(self.X_train_bal)
            self.X_test_scaled=self.sc.transform(self.X_test)
            #training variables self.X_train_bal_scaled,self.y_train_bal
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
    def all_model(self):
        try:
            common(self.X_train_bal_scaled,self.y_train_bal,self.X_test_scaled,self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
    def best_model(self):
        try:
            nb_obj=GaussianNB(var_smoothing=np.float64(1e-12))
            nb_obj.fit(self.X_train_bal_scaled,self.y_train_bal)
            logger.info(f"accuracy score:{accuracy_score(self.y_test,nb_obj.predict(self.X_test_scaled))}")
            logger.info(f"confusion matrix:{confusion_matrix(self.y_test,nb_obj.predict(self.X_test_scaled))}")
            logger.info(f"classification report:{classification_report(self.y_test,nb_obj.predict(self.X_test_scaled))}")
            #hyperparamter tuning
            # parameter_list={
            #     "var_smoothing":np.logspace(-12,-1,12)
            # }
            # grid_obj=GridSearchCV(estimator=nb_obj,param_grid=parameter_list,cv=10,scoring="accuracy",n_jobs=-1)
            # grid_obj.fit(self.X_train_bal_scaled,self.y_train_bal)
            # logger.info(f"best parameters:{grid_obj.best_params_}")
            # logger.info(f"best score:{grid_obj.best_score_}")
            prediction=np.array([[63,1,3,150,2.3,0,1]])
            self.sc.transform(prediction)
            logger.info(nb_obj.predict(prediction)[0])
            with open('model.pkl', 'wb') as f:
                pickle.dump(nb_obj, f)
            with open('scaled.pkl', 'wb') as p:
                pickle.dump(self.sc, p)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
if __name__ == "__main__":
    try:
        obj=HEART("heart.csv")
        obj.vt_outliers()
        obj.feature_selection()
        obj.data_balancing()
        obj.all_model()
        obj.best_model()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
