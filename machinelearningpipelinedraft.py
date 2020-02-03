# importing packages
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
import imblearn
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# reading in CSV
df = pd.read_csv('creditcard.csv')
df.sample(5)

df.info()

df.describe()

# taking a closer look at the class variable
sns.countplot('Class', data = df)
plt.title('No Fraud (0) vs. Fraud (1)')

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
import math
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

class modelpipeline:
    def __init__(self):
        pass
    
    def run_model(self, df, testratio, standardize, sampletype, modelname, text):
        df = df.drop('Time', axis=1)
        if standardize == True:
            df = self.standardize(df)
        if sampletype == 'smote':
            X_train, X_test, y_train, y_test = sampling.smote_oversample(df, testratio)
        elif sampletype == 'adasyn':
            X_train, X_test, y_train, y_test = sampling.adasyn_oversample(df, testratio)
        else:
            X_train, X_test, y_train, y_test = sampling.naive_oversample(df, testratio)
        store = self.build_model(X_train, X_test, y_train, y_test, text, modelname)
        # test model with all actual fraud results
        store['actual_accuracy'] = evaluate.actual_acc(df, store['model'])
        return store
    
    def build_model(self, X_train, X_test, y_train, y_test, text, modelname):
        if modelname == 'LogisticRegression':
            param_grid = dict(C=[0.8,0.9,1,1.1], max_iter=[300], solver='liblinear')
            LogRegression = LogisticRegression()
            model = GridSearchCV(LogRegression, param_grid, cv=10, scoring='f1')
            model.fit(X_train,y_train)
            print("Best f1 score: " + str(model.best_score_))
            print("Best parameters: " + str(model.best_params_))
        elif modelname == 'XGBoost':
            model = xgb.XGBClassifier(seed=42, nthread=1, max_depth=math.ceil(math.sqrt(X_train.shape[1])),
                                      n_estimators=100, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=5)
        elif modelname == 'RandomForest':
            start_value = math.ceil(math.sqrt(X_train.shape[1]))
            end_value = start_value + 5          
            treedepth = list(range(start_value, end_value, 2))
            param_grid = dict(random_state=[42], max_depth=treedepth, n_estimators=[100,150])
            RFC = RandomForestClassifier()
            model = GridSearchCV(RFC, param_grid, cv=10, scoring='f1')
            model.fit(X_train,y_train)
            print("Best f1 score: " + str(model.best_score_))
            print("Best parameters: " + str(model.best_params_))                      
        else:
            model = LogisticRegression()
            model.fit(X_train,y_train)
        y_predict = model.predict(X_test)
        results = evaluate.model_results(y_test, y_predict, text)
        store = {"model": model, "X_train": X_train, "X_test": X_test, "y_train": y_train, 
                 "y_test": y_test, "results": results}
        print("Model fitting and results are complete!")
        return store
    
    def standardize(self, df):
        columns = df.columns.values.tolist()
        columns.remove('Class')
        for column in columns:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
        return df

class sampling:
    def __init__(self):
        pass
    @staticmethod
    def naive_oversample(df, testratio):
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testratio, random_state=42)
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        # train test split keeps X_test and y_test as pd series, oversampler converts X_train, y_train to numpy
        # Convert all to numpy array for XGBoost to not have bugs
        X_test = X_test.values
        y_test = y_test.values
        print("Oversampling is complete!")
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def smote_oversample(df, testratio):
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testratio, random_state=42)
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        # train test split keeps X_test and y_test as pd series, oversampler converts X_train, y_train to numpy
        # Convert all to numpy array for XGBoost to not have bugs
        X_test = X_test.values
        y_test = y_test.values
        print("Number of Xs and Ys for SMOTE:")
        print(sorted(Counter(y_train).items()))
        print("Oversampling is complete!")
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def adasyn_oversample(df, testratio):
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testratio, random_state=42)
        X_train, y_train = ADASYN().fit_resample(X_train, y_train)
        # train test split keeps X_test and y_test as pd series, oversampler converts X_train, y_train to numpy
        # Convert all to numpy array for XGBoost to not have bugs
        X_test = X_test.values
        y_test = y_test.values
        print("Number of Xs and Ys for ADASYN:")
        print(sorted(Counter(y_train).items()))
        print("Oversampling is complete!")
        return X_train, X_test, y_train, y_test



class evaluate:
    def __init__(self):
        pass
    
    @staticmethod
    def model_results(y_test, y_predict, text):
        cm = metrics.confusion_matrix(y_test, y_predict)
        print(cm)
        RFC_CM = pd.DataFrame(cm, ['Actual 0', 'Actual 1'], ['Predict 0', 'Predict 1'])
        sns.heatmap(RFC_CM, annot=True, annot_kws={"size": 16}, cmap='Greens', linewidths=1, fmt='g')# font size
        sns.set(font_scale=1.4)#for label size
        plt.title("Confusion Matrix for " + text)

        # fix for mpl bug that cuts off top/bottom of seaborn viz
        b, t = plt.ylim() 
        b += 0.5 
        t -= 0.5 
        plt.ylim(b, t) 
        plt.show() 

        accuracy = metrics.accuracy_score(y_test, y_predict)
        print('Accuracy: ' + str(accuracy))
        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
        recall = sensitivity
        print('Sensitivity: ' + str(sensitivity))
        specificity = cm[0][0] / (cm[0][0] + cm[0][1])
        print('Specificity: ' + str(specificity))
        precision = cm[1][1] / (cm[1][1] + cm[0][1])
        print('Precision: ' + str(precision))
        f1 = 2 * (recall * precision)/(recall + precision)
        print('f1 score: ' + str(f1))
        auc = evaluate.ROC(y_test, y_predict, text)
        results = {"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity, 
                   "precision": precision, "f1": f1, "auc": auc}
        print("Model classification metrics have finished calculating!")
        return results
    
    @staticmethod
    def ROC(y_test, y_predict, text):
        # IMPORTANT: first argument is true values, second argument is predicted probabilities
        auc = metrics.roc_auc_score(y_test, y_predict)
        print("AUC value is: " + str(auc))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title('ROC curve for ' + text)
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True)
        return auc

    @staticmethod
    def actual_acc(df, model):
        allpositive = df[df['Class'] == 1].copy()
        x_positive = allpositive.drop(['Class'], axis=1)
        y_positive = allpositive['Class']
        y_pospredict = model.predict(x_positive)
        accuracy_positive = metrics.accuracy_score(y_positive, y_pospredict)
        print("Accuracy with all fraud results is " + str(accuracy_positive * 100) + "%")
        return accuracy_positive
