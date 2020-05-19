import numpy as np
import pandas as pd
import operator
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
import statsmodels.discrete.discrete_model as sm


def forward_selection(df, sig_level, response, removelist, sampling='nil', testratio=0):
    """
    :param df: dataframe with both training and response variables
    :param sig_level: significance level to accept/reject var during forward selection
    :param response: name of response var in dataframe
    :param removelist: list of training variables to remove from dataframe
    :param sampling: type of oversampling to use, smote, naive or nil, default: no sampling done
    :param testratio: proportion of dataset to remove out before doing oversampling, default: 0
    :return: list of training variables, actualvars
    """
    
    if isinstance(removelist, str) == True:
        temp_str = removelist
        internallist = []
        internallist.append(temp_str)
    else:
        internallist = removelist
    X = df.drop(internallist, axis=1)
    y = df[response]
    # Get list of column names
    colnames = list(X.columns.values)
    print(colnames)

    # Start of train-test split and oversampling (if relevant)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testratio, random_state=42)
    if sampling.lower() == 'smote':
        print("SMOTE Oversampling selected..")
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        # train test split keeps X_test and y_test as pd series, oversampler converts X_train, y_train to numpy
        # Convert all to numpy array for XGBoost to not have bugs
        X_test = X_test.values
        y_test = y_test.values
        print("Number of Xs and Ys for: " + str(sampling.upper()))
        print(sorted(Counter(y_train).items()))
        print("Oversampling is complete!")
    elif sampling.lower() == 'naive':
        print("Naive Oversampling selected..")
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        # train test split keeps X_test and y_test as pd series, oversampler converts X_train, y_train to numpy
        # Convert all to numpy array for XGBoost to not have bugs
        X_test = X_test.values
        y_test = y_test.values
        print("Number of Xs and Ys for: " + str(sampling.upper()))
        print(sorted(Counter(y_train).items()))
        print("Oversampling is complete!")
    else:
        print("No sampling selected..")

    # Total features to select = k
    # In each iteration, the current set of n features is concatenated with a new feature not inside current set
    # It is then sent for training with the logistic regression
    # The model performance for each feature + current features is evaluated by its highest p value (worst feature)
    # All highest p values of all feature addition to n features (k-n iterations) are put into a dictionary
    # Next, the lowest p value out of all the iterations (for n features + 1) is chosen for evaluation
    # Set significance level, which is compared to the lowest p value of the best model in the current training iteration
    # If best model in current training iteration of n vars has any vars with p value > sig level, then the model training stops
    # Because all the different models are worse or equally bad as the current best model, we can terminate selection process
    # If not, repeat this iteration with now n+1 features and k-n-1 iterations

    maxcolsnum = X_train.shape[1]
    full_x = np.array(False)
    allowed_nums = {}
    for i in range(maxcolsnum):
        allowed_nums[i] = True
    actual_nums = []
    actual_vars = []
    terminate_early = False
    y = y_train
    for i in range(maxcolsnum):
        # Reset boolean and pval_list
        terminate_early = False
        pval_list = {}
        for j in range(maxcolsnum):
            if allowed_nums[j] == True:
                # Need to reshape to single column instead of a long array for concating properly
                jth_x = X_train[:, j].reshape(-1, 1)
                if full_x.any():
                    iter_x = np.concatenate((full_x, jth_x), axis=1)
                else:
                    iter_x = jth_x
                regressor_OLS = sm.Logit(y_train, iter_x).fit(disp=0)
                pval_list[j] = max(regressor_OLS.pvalues)
                # Special condition where all the features have p values of 0, directly use these variables for training
                if max(regressor_OLS.pvalues) == 0:
                    if full_x.any():
                        full_x = np.concatenate((full_x, jth_x), axis=1)
                        allowed_nums[j] = False
                        actual_nums.append(j)
                        print("Features all have p value of 0, using feature: [" + str(colnames[j]) + "]")
                    else:
                        full_x = jth_x
                        allowed_nums[j] = False
                        actual_nums.append(j)
                        print("First model trained using feature: [" + str(colnames[j]) + "] with p value of 0")
                    terminate_early = True
                    break
            else:
                continue
        if i > 0 and terminate_early == False:
            print("Building new model with lowest p-values with " + str(len(actual_nums)) + " variables.")
            max_pval_col = min(pval_list.items(), key=operator.itemgetter(1))[0]
            max_pval = pval_list[max_pval_col]
            # Need to reshape to single column instead of a long array for concating properly
            jth_x = X_train[:, max_pval_col].reshape(-1, 1)
            if max_pval < sig_level:
                if full_x.any():
                    full_x = np.concatenate((full_x, jth_x), axis=1)
                    allowed_nums[max_pval_col] = False
                    actual_nums.append(max_pval_col)
                    print("New model trained using feature: [" + str(
                        colnames[max_pval_col]) + "] with lowest p values of " + str(max_pval))
                else:
                    full_x = jth_x
                    allowed_nums[max_pval_col] = False
                    actual_nums.append(max_pval_col)
                    print("First model trained using feature: [" + str(
                        colnames[max_pval_col]) + "] with lowest p values of " + str(max_pval))
            else:
                print("TERMINATING AS best model trained using feature: [" + str(
                    colnames[max_pval_col]) + "] with high p value of " + str(
                    max_pval) + " above significance level: " + str(sig_level))
                break

    for k in actual_nums:
        actual_vars.append(colnames[k])
    print('Final variables selected:')
    print(actual_vars)

    return actual_vars