# importing packages
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
import imblearn
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb 
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
import math
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from collections import Counter

class modelpipeline:
    def __init__(self):
        pass
    
    def run_model(self, df, varlist, response, testratio, standardize, sampletype, modelname, text, CV):
        # Align field orders in df (including response)
        df = df[varlist]

        if sampletype == 'smote':
            X_train, X_test, y_train, y_test = sampling.smote_oversample(df, testratio, response)
        elif sampletype == 'adasyn':
            X_train, X_test, y_train, y_test = sampling.adasyn_oversample(df, testratio, response)
        else:
            X_train, X_test, y_train, y_test = sampling.naive_oversample(df, testratio, response)
            
        if standardize == True:
            scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
            X_train = scaling.transform(X_train)
            X_test = scaling.transform(X_test)
            
        store = self.build_model(X_train, X_test, y_train, y_test, text, modelname, CV)
        # test model with all actual fraud results
        store['actual_accuracy'] = evaluate.actual_acc(df, store['model'], response)
        return store
    
    def build_model(self, X_train, X_test, y_train, y_test, text, modelname, CV):
        if modelname == 'LogisticRegression':
            if CV == True:
                param_grid = dict(C=[0.8,1,1.2], max_iter=[300], solver=['liblinear'])
                LogRegression = LogisticRegression()
                model = GridSearchCV(LogRegression, param_grid, cv=5, scoring='f1', verbose=10)
                model.fit(X_train,y_train)
                print("Best f1 score: " + str(model.best_score_))
                print("Best parameters: " + str(model.best_params_))
            else:
                model = LogisticRegression(max_iter=300, C=0.8, solver='liblinear')
                model.fit(X_train,y_train)
        elif modelname == 'XGBoost':
            if CV == True:
                end_value = math.ceil(math.sqrt(X_train.shape[1]))
                start_value = end_value - 2       
                # treedepth = list(range(start_value, end_value+1, 2))
                param_grid = dict(n_estimators=[100], max_depth=[end_value])
                GradientBoost = GradientBoostingClassifier()
                model = GridSearchCV(GradientBoost, param_grid, cv=5, scoring='f1', verbose=10)
                model.fit(X_train,y_train)
                print("Best f1 score: " + str(model.best_score_))
                print("Best parameters: " + str(model.best_params_))
                
                # Testing out xgb.cv (incomplete)
                # model = xgb.XGBClassifier(seed=42, nthread=1, max_depth=start_value, n_estimators=100, random_state=42)
                # xgb_param = dict(n_estimators=100, max_depth=end_value)
                # xgtrain = xgb.DMatrix(X_train, label=y_train)
                # model = xgb.cv(params=xgb_param, dtrain=xgtrain, nfold=5, metrics='auc')
                # model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=5)
                
                # USING kfold library to do kfold testing on XGBoost:
                # cross_val_score using kfold does not fit the model, so nothing can be predicted
                # it's just to see the results but the model has to be fitted later on
                # kfold = KFold(n_splits=3, random_state=42)
                # print(kfold)
                # scores = cross_val_score(model, X_train, y_train, cv=kfold)
                # print("CV Accuracy: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))
            else:
                model = xgb.XGBClassifier(seed=42, nthread=1, max_depth=math.ceil(math.sqrt(X_train.shape[1])),
                                          n_estimators=100, random_state=42)
                model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=5)
        elif modelname == 'RandomForest':
            if CV == True:
                start_value = math.ceil(math.sqrt(X_train.shape[1]))
                end_value = start_value + 11         
                treedepth = list(range(start_value, end_value, 5))
                param_grid = dict(random_state=[42], max_depth=treedepth, n_estimators=[100,150])
                RFC = RandomForestClassifier()
                model = GridSearchCV(RFC, param_grid, cv=5, scoring='f1', verbose=10)
                model.fit(X_train,y_train)
                print("Best f1 score: " + str(model.best_score_))
                print("Best parameters: " + str(model.best_params_))
            else:
                treedepth = math.ceil(math.sqrt(X_train.shape[1]))
                model = RandomForestClassifier(random_state=42, max_depth=treedepth, n_estimators=150)
                model.fit(X_train,y_train)
        else:
            # Parameters based on gridsearchcv of modelname = logistic regresion
            # Leave parameter blank for modelname to run this instance of logistic regression
            model = LogisticRegression(C=0.8, max_iter=300, solver='liblinear')
            model.fit(X_train,y_train)
        
        y_predict = model.predict(X_test)
        y_predictprob = model.predict_proba(X_test)[:, 1]
        results = evaluate.model_results(y_test, y_predict, y_predictprob, text)
        store = {"model": model, "X_train": X_train, "X_test": X_test, "y_train": y_train, 
                 "y_test": y_test, "results": results}
        print("Model fitting and results are complete!")
        return store
    
    def standardize(self, df):
        # Variables already standardized except for Amount
        # columns = df.columns.values.tolist()
        # columns.remove(response)
        for column in ['Amount']:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
        return df

class sampling:
    def __init__(self):
        pass
    @staticmethod
    def naive_oversample(df, testratio, response):
        X = df.drop([response], axis=1)
        y = df[response]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testratio, random_state=41)
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        # train test split keeps X_test and y_test as pd series, oversampler converts X_train, y_train to numpy
        # Convert all to numpy array for XGBoost to not have bugs
        X_test = X_test.values
        y_test = y_test.values
        print("Oversampling is complete!")
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def smote_oversample(df, testratio, response):
        X = df.drop([response], axis=1)
        y = df[response]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testratio, random_state=41)
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
    def adasyn_oversample(df, testratio, response):
        X = df.drop([response], axis=1)
        y = df[response]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testratio, random_state=41)
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
    def model_results(y_test, y_predict, y_predictprob, text):
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
        auc, pr_auc = evaluate.ROC(y_test, y_predictprob, text)
        results = {"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity, 
                   "precision": precision, "f1": f1, "auc": auc, "pr_auc": pr_auc}
        print("Model classification metrics have finished calculating!")
        print(results)
        return results
    
    @staticmethod
    def ROC(y_test, y_predictprob, text):
        # IMPORTANT: first argument is true values, second argument is predicted probabilities
        auc = metrics.roc_auc_score(y_test, y_predictprob)
        # print("AUC value is: " + str(auc))
        print("AUC value is: " + str(auc))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predictprob)
        # print("AUC value is also: " + str(metrics.auc(fpr, tpr)))
        # Calculate precision and recall for each threshold
        precision, recall, _ = metrics.precision_recall_curve(y_test, y_predictprob)
        pr_auc = metrics.auc(recall, precision)
        fullgraph = plt.figure(1,figsize=(10,20))
        plt.style.use('ggplot')

        ROCAUC_plot = fullgraph.add_subplot(211)
        ROCAUC_plot.plot(fpr, tpr, color='blue')
        ROCAUC_plot.set_title('ROC curve for ' + text)
        ROCAUC_plot.set_xlabel('False Positive Rate (1 - Specificity)')
        ROCAUC_plot.set_ylabel('True Positive Rate (Sensitivity)')
        ROCAUC_plot.set_xlim([0.0, 1.0])
        ROCAUC_plot.set_ylim([0.0, 1.0])
        ROCAUC_plot.grid(True)
        PRAUC_plot = fullgraph.add_subplot(212)
        PRAUC_plot.plot(precision, recall, color='purple')
        PRAUC_plot.set_title('Precision-Recall curve for ' + text)
        PRAUC_plot.set_xlabel('Recall')
        PRAUC_plot.set_ylabel('Precision')
        PRAUC_plot.set_xlim([0.0, 1.0])
        PRAUC_plot.set_ylim([0.0, 1.0])
        PRAUC_plot.grid(True)
        return auc, pr_auc

    @staticmethod
    def actual_acc(df, model, response):
        allpositive = df[df[response] == 1].copy()
        x_positive = allpositive.drop([response], axis=1)
        y_positive = allpositive[response]
        # Convert to numpy array due to XGBoost model.predict not working well for pandas
        x_positive = x_positive.values
        y_positive = y_positive.values
        y_pospredict = model.predict(x_positive)
        accuracy_positive = metrics.accuracy_score(y_positive, y_pospredict)
        print("Accuracy with all fraud results is " + str(accuracy_positive * 100) + "%")
        return accuracy_positive