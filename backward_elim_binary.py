import statsmodels.discrete.discrete_model as sm


def backward_elimination(x, Y, sl, columns):
    """
    :param x: numpy array of training variables
    :param Y: Y is numpy array of response variable
    :param sl: significance level in float
    :param columns: list of columns in same horizontal order with x
    :return: numpy array of selected x, list of selected training variables passing sig level
    """
    numVars = len(x[0])  # Get length of a row for num of vars
    for i in range(0, numVars):
        regressor_OLS = sm.Logit(Y, x, maxiter=200).fit()
        # for loop to fit current set of all vars in x and response Y
        # As loop goes on, x gets smaller as columns are deleted, and columns keeping track of col names also edited
        maxVar = max(regressor_OLS.pvalues).astype(float)
        print('Regression model retrained with ' + str(numVars) + ' variables.')
        print('Max p value for a feature is: ' + str(maxVar))
        # get max p value and if its more than sig level, start deleting jth column
        # Since columns are getting deleted and x gets smaller, we need to update columns keeping track of column names
        # Hence the only way to ensure the deletion of right column
        # is to check the max p value with the current pvalues[j] in regression model
        # if they are same, then jth column safe to delete
        if maxVar > sl:
            print('Max p value > ' + str(sl) + ', feature will be removed.')
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    print(str(j) + 'th column deleted: ' + str(columns[j]))
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    numVars -= 1
        else:
            print('All p values are above ' + str(sl) + '. Terminating model training')
            print('p values list: ' + str(regressor_OLS.pvalues))
            break

    print(regressor_OLS.summary())
    return x, columns  # Return x data and list of columns