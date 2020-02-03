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
from collections import Counter

# To run pipeline, create new class object modelpipeline()
# Next, execute modelpipeline.runmodel(...)
# Parameters are:
# df - DataFrame that has went through data cleaning, processing, feature engineering, containing both features and response
# No standardization/scaling is required as there is built in function for that
# varlist - List of all variables/features to use in model, including the response variable. 
# This allows user to do feature selection first and select only the relevant features before running the model
# response - Name of the response variable in string format
# sampletype - For undersampled data, you can use 'naive', 'smote' or 'adasyn' oversampling. If other strings are input, then no oversampling is done.
# modelname - Choose the type of model to run - user can add more models as required using the if/else clause to check this string input in the buildmodel function
# text - Text title to put for the confusion matrix output in each iteration of the n-folds stratified cross validation
# n-fold - number of folds for the stratified cross validation

class modelpipeline:
    def __init__(self):
        pass
    
    def run_model(self, df, varlist, response, standardize, sampletype, modelname, text, n_fold):
        # Remove any features not wanted based on varlist input
        df = df[varlist]
        # We have to remove response from varlist as it is used later to subset out features
        # Refer to the for loop for the cross validation where X_train and X_test is created at the end of loop
        varlist.remove(response)
            
        if isinstance(n_fold, int) and n_fold > 1:
            # Initialize dictionary to store results
            self.store = {"accuracy": [], "actual_accuracy": [], "sensitivity": [], "specificity": [], 
                          "precision": [], "f1": [], "auc": [], "final": {}}
            
            # Split dataframes into 2, one for positive response and one for negative
            df_zero = df[df[response] == 0]
            df_one = df[df[response] == 1]
            
            # Shuffle dataframe for response=0 and =1 so that train-test will not be biased in case rows that are similar are placed side by side
            # Later on, we will reset the index and select by the index number by sections
            df_zero = shuffle(df_zero, random_state=42)
            df_one = shuffle(df_one, random_state=42)
            df_zero = df_zero.reset_index(drop=True)
            df_one = df_one.reset_index(drop=True)
        
            # Get the average number of records required for negative response and positive response for test records
            # Train records will then have all the other records not in the test records
            # n_fold is the number of folds for cross validation
            start_index_one = 0
            end_index_one = math.floor(df_one.shape[0]/n_fold)
            start_index_zero = 0
            end_index_zero = math.floor(df_zero.shape[0]/n_fold)
            
            for i in range(1,n_fold+1):
                if i != n_fold:
                    print('Getting TEST DF for response 1 from index ' + str(start_index_one) + ' to ' + str(end_index_one))
                    df_one_test = df_one.iloc[start_index_one:end_index_one]
                    print('Getting TRAIN DF for response 1 from index 0 to ' + str(start_index_one) + ' and from index ' + str(end_index_one) + ' to ' + str(df_one.shape[0]))
                    df_one_train = pd.concat([df_one.iloc[0:start_index_one],df_one.iloc[end_index_one:]], axis=0)
                    start_index_one += math.floor(df_one.shape[0]/n_fold)
                    end_index_one += math.floor(df_one.shape[0]/n_fold)
                    
                    print('Getting TEST DF for response 0 from index ' + str(start_index_zero) + ' to ' + str(end_index_zero))
                    df_zero_test = df_zero.iloc[start_index_zero:end_index_zero]
                    print('Getting TRAIN DF for response 0 from index 0 to ' + str(start_index_zero) + ' and from index ' + str(end_index_zero) + ' to ' + str(df_zero.shape[0]))
                    df_zero_train = pd.concat([df_zero.iloc[0:start_index_zero],df_zero.iloc[end_index_zero:]], axis=0)
                    start_index_zero += math.floor(df_zero.shape[0]/n_fold)
                    end_index_zero += math.floor(df_zero.shape[0]/n_fold)

                else:
                    # Last section of split needs to reach the end of dataset
                    print('Getting TEST DF for response 1 from index ' + str(start_index_one) + ' to ' + str(df_one.shape[0]))
                    df_one_test = df_one.iloc[start_index_one:df_one.shape[0]]
                    print('Getting TRAIN DF for response 1 from index 0 to ' + str(start_index_one))
                    df_one_train = df_one.iloc[0:start_index_one]
                    
                    # Last section of split needs to reach the end of dataset
                    print('Getting TEST DF for response 0 from index ' + str(start_index_zero) + ' to ' + str(df_zero.shape[0]))
                    df_zero_test = df_zero.iloc[start_index_zero:df_zero.shape[0]]
                    print('Getting TRAIN DF for response 0 from index 0 to ' + str(start_index_zero))
                    df_zero_train = df_zero.iloc[0:start_index_zero]
                    
                # Combine the subsetted sections for negatives and postives for both train and test before oversampling  
                df_train = pd.concat([df_one_train, df_zero_train], axis=0)
                df_test = pd.concat([df_one_test, df_zero_test], axis=0)
                # varlist has the feature list X without Y while response is the Y
                # print(varlist)
                X_train = df_train[varlist]
                # print('Check X train vars after combining pds')
                # print(X_train.columns.values)
                y_train = df_train[response]
                X_test = df_test[varlist]
                y_test = df_test[response]
                
                if standardize == True:
                    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
                    X_train = scaling.transform(X_train)
                    X_test = scaling.transform(X_test)
                    X_train = pd.DataFrame(X_train, columns=varlist)
                    X_test = pd.DataFrame(X_test, columns=varlist)

                if sampletype == 'smote':
                    X_train, X_test, y_train, y_test = sampling.smote_oversample(X_train, X_test, y_train, y_test, response)
                elif sampletype == 'adasyn':
                    X_train, X_test, y_train, y_test = sampling.adasyn_oversample(X_train, X_test, y_train, y_test, response)
                elif sampletype == 'naive':
                    X_train, X_test, y_train, y_test = sampling.naive_oversample(X_train, X_test, y_train, y_test, response)
                else:
                    # Convert all DF to numpy array for model building later
                    X_train = X_train.values
                    y_train = y_train.values
                    X_test = X_test.values
                    y_test = y_test.values
                
                # Build model in current fold/iteration and get accuracy, sensitivity, specificity, precision, f1, auc
                self.store = self.build_model(X_train, X_test, y_train, y_test, text, modelname, i, n_fold, self.store)
                
                # test model with all actual fraud results
                self.store['actual_accuracy'].append(evaluate.actual_acc(df, self.store['model'], response))
                
            # Before results are returned, get average of all evaluation metrics and store in store['final'] section
            self.store['final']['accuracy'] = self.avg(self.store['accuracy'])
            self.store['final']['sensitivity'] = self.avg(self.store['sensitivity'])
            self.store['final']['specificity'] = self.avg(self.store['specificity'])
            self.store['final']['precision'] = self.avg(self.store['precision'])
            self.store['final']['f1'] = self.avg(self.store['f1'])
            self.store['final']['auc'] = self.avg(self.store['auc'])
            self.store['final']['actual_accuracy'] = self.avg(self.store['actual_accuracy'])
            
            # Put back response that was removed previously
            varlist.append(response)
            
            print('Final Results of ' + str(n_fold) + ' fold CV:')
            print(self.store['final'])
            return self.store
        
        else:
            print('n fold must be an integer greater than 1')
            
            # Put back response that was removed previously
            varlist.append(response)
            return self.store
    
    def build_model(self, X_train, X_test, y_train, y_test, text, modelname, i, n_fold, store):
        if modelname == 'LogisticRegression':
            model = LogisticRegression(max_iter=300, C=0.8, solver='liblinear')
            model.fit(X_train,y_train)
        elif modelname == 'XGBoost':
            model = xgb.XGBClassifier(seed=42, nthread=1, max_depth=math.ceil(math.sqrt(X_train.shape[1])),
                                      n_estimators=100, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=5)
        elif modelname == 'XGBoostminus1':
            # XGBoost with one less depth
            model = xgb.XGBClassifier(seed=42, nthread=1, max_depth=math.ceil(math.sqrt(X_train.shape[1])-1),
                                      n_estimators=100, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=5)
        elif modelname == 'XGBoostplus1':
            # XGBoost with one less depth
            model = xgb.XGBClassifier(seed=42, nthread=1, max_depth=math.ceil(math.sqrt(X_train.shape[1]))+1,
                                      n_estimators=100, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=5)
        elif modelname == 'SVM_Linear':
            model = LinearSVC(random_state=42) # default C=1 (regularization parameter)
            model.fit(X_train,y_train)
        elif modelname == 'SVM_Linear2':
            model = LinearSVC(random_state=42, C=2) 
            model.fit(X_train,y_train)
        elif modelname == 'SVM_Linear0.5':
            model = LinearSVC(random_state=42, C=0.5)
            model.fit(X_train,y_train)
        elif modelname == 'SVM_Linear0.3':
            model = LinearSVC(random_state=42, C=0.3)
            model.fit(X_train,y_train)
        elif modelname == 'RandomForest':
            treedepth = math.ceil(math.sqrt(X_train.shape[1]))
            model = RandomForestClassifier(random_state=42, max_depth=treedepth, n_estimators=100)
            model.fit(X_train,y_train)
        else:
            # Parameters based on gridsearchcv of modelname = logistic regresion
            # Leave parameter blank for modelname to run this instance of logistic regression
            model = LogisticRegression(C=0.8, max_iter=300, solver='liblinear')
            model.fit(X_train,y_train)
        
        y_predict = model.predict(X_test)
        store = evaluate.model_results(y_test, y_predict, text, store)
        
        # Store model for usage in measuring actual accuracy of fraud cases
        store['model'] = model
        print("Iteration " + str(i) + " out of " + str(n_fold) + " of CV for model fitting and obtaining results is complete!")
        print("\n")
        return store
    
#     def standardize(self, df, varlist, response):
#         x_values = df[varlist]
#         y_values = df[response]
#         scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_values)
#         x_values = scaling.transform(x_values)
#         x_values = pd.DataFrame(x_values, columns=varlist)
#         df = pd.concat([x_values,y_values], axis=1)
        # Variables already standardized except for Amount
        # columns = df.columns.values.tolist()
        # columns.remove(response)
        # for column in varlist:
        #     df[column] = (df[column] - df[column].mean()) / df[column].std()

        return df
    
    def avg(self, array):
        return sum(array) / len(array)

class sampling:
    def __init__(self):
        pass
    @staticmethod
    def naive_oversample(X_train, X_test, y_train, y_test, response):
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        # train test split keeps X_test and y_test as pd series, oversampler converts X_train, y_train to numpy
        # Convert all to numpy array for XGBoost to not have bugs
        X_test = X_test.values
        y_test = y_test.values
        print("Oversampling is complete!")
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def smote_oversample(X_train, X_test, y_train, y_test, response):
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
    def adasyn_oversample(X_train, X_test, y_train, y_test, response):
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
    def model_results(y_test, y_predict, text, store):
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
        # print('Accuracy: ' + str(accuracy))
        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
        recall = sensitivity
        # print('Sensitivity: ' + str(sensitivity))
        specificity = cm[0][0] / (cm[0][0] + cm[0][1])
        # print('Specificity: ' + str(specificity))
        precision = cm[1][1] / (cm[1][1] + cm[0][1])
        # print('Precision: ' + str(precision))
        f1 = 2 * (recall * precision)/(recall + precision)
        # print('f1 score: ' + str(f1))
        auc = evaluate.ROC(y_test, y_predict, text)
        
        store['accuracy'].append(accuracy)
        store['sensitivity'].append(sensitivity)
        store['specificity'].append(specificity)
        store['precision'].append(precision)
        store['f1'].append(f1)
        store['auc'].append(auc)

        return store
    
    @staticmethod
    def ROC(y_test, y_predict, text):
        # IMPORTANT: first argument is true values, second argument is predicted probabilities
        auc = metrics.roc_auc_score(y_test, y_predict)
        # print("AUC value is: " + str(auc))
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
    def actual_acc(df, model, response):
        allpositive = df[df[response] == 1]
        x_positive = allpositive.drop([response], axis=1)
        y_positive = allpositive[response]
        # Convert to numpy array due to XGBoost model.predict not working well for pandas
        x_positive = x_positive.values
        y_positive = y_positive.values
        y_pospredict = model.predict(x_positive)
        accuracy_positive = metrics.accuracy_score(y_positive, y_pospredict)
        # print("Accuracy with all fraud results is " + str(accuracy_positive * 100) + "%")
        return accuracy_positive