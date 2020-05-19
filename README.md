# Fraud-Detection-Pipeline
A structured data pipeline for classification problems that does multiple purposes like scaling, sampling, k-fold stratified cross validation (CV) with evaluation metrics. <br>
It reduces the need for users to rewrite a lot of code as it's reusability is very high.<br>
Refer to <b>sklearn_classification_pipeline.py</b> for the full code <br>
For pipeline that does not have k-fold cross validation which leads to faster testing, use sklearn_classifier_pipeline_optionalCV.py. However this file still supports the internal sklearn cross validation method (can be switched on or off by parameter input). To use the custom k-fold stratified cross validation method, use sklearn_classification_pipeline.py instead.
<br><br>
# Strengths of using a data pipeline:
1. Customized pipeline works for all forms of classification problems, including fraud detection problems that require oversampling techniques. <br>
2. Data pipeline allows users to do prior data cleansing and feature engineering, as long as df is DataFrame format with both features and response <br>
3. As users might have a list of features to select, pipeline allows a varlist input which is an array of features to use for the DF. This complements any feature selection techniques used before this pipeline such as forward/backward selection <br>
4. Data pipeline also caters for stratified k-fold cross validation for any value k>1 <br>
5. Pipeline supports various kinds of evaluation metrics (accuracy, sensitivity/recall, precision, selectivity, f1, AUC value) and user can add more as required <br>
6. User can easily add new models into the pipeline as required without rerunning/rewriting a lot of code <br>
7. Pipeline allows for data scaling/standardization <br>
8. High customizability and reusability of the code while reducing the need to rewrite a lot of code (leading to spaghetti coding)

# Instructions:
To run pipeline, import sklearn_classification_pipeline.py (stratified CV) or sklearn_classifier_pipeline_optionalCV.py (can switch off CV). <br>
Create new class object modelpipeline() <br>
Next, execute modelpipeline.runmodel(...) with the required parameters <br>

## Parameters for sklearn_classification_pipeline.py are:
1. df - DataFrame that has went through data cleaning, processing, feature engineering, containing both features and response
<br> No standardization/scaling is required as there is built in function for that <br>
2. varlist - List of all variables/features to use in model, including the response variable. <br>
<br> This allows user to do feature selection first and select only the relevant features before running the model <br>
3. response - Name of the response variable in string format <br>
4. sampletype - For undersampled data, you can use 'naive', 'smote' or 'adasyn' oversampling. If other strings are input, then no oversampling is done. <br>
5. modelname - Choose the type of model to run - user can add more models as required using the if/else clause to check this string input in the buildmodel function <br>
6. text - Text title to put for the confusion matrix output in each iteration of the n-folds stratified cross validation <br>
7. n-fold - number of folds for the stratified cross validation <br>
8. Note that sklearn_classifier_pipeline_optionalCV.py has an additional parameter at the end called CV. If CV=False, then cross validation will be switched off. <br>
9. Remember to save the dictionary object into a variable - e.g. results = modelpipeline.runmodel(...) so that the evaluation results can be saved and reused.

# Results:
![Confusion Matrix returned from each iteration of k-fold cross validation](https://github.com/kohjiaxuan/Fraud-Detection-Pipeline/blob/master/Confusion_Matrix.PNG)
<br><br>
After the tests have finished, you can read the dictionary object storing the evaluation metrics results. In this case, results['final'] store the averaged results for k-fold cross validation while the other key-value pairs will store the evaluation metric result of each individual iteration in a list. <br>
![Sample Results returned from the modelpipeline object](https://github.com/kohjiaxuan/Fraud-Detection-Pipeline/blob/master/results.PNG)
<br>
The dictionary object returned will have results for each fold of k-fold cross validation. Evaluation metrics include: <br>
1. Accuracy
2. Actual Accuracy (Optional - can be a hold out dataset for testing and can be other metrics other than accuracy)
3. Sensitvity
4. Specificity
5. Precision
6. f1 score
7. (ROC) AUC value
8. (PR) AUC value
9. Averaged values for 1-8 stored in dictionary object tagged to 'final' key
<br>
Object Template = {"accuracy": [...], "actual_accuracy": [...], "sensitivity": [...], "specificity": [...], 
                          "precision": [...], "f1": [...], "auc": [...], "pr_auc": [...], "final": {...}}
<br>
For sklearn_classifier_pipeline_optionalCV.py, it returns results in a string format instead of a list of strings as there is only round of train-test. There is also the option to tweak the code to export the best model/transformed train-test dataset out for usage. By default, this is not done to free up memory usage. <br><br>

## Graphs for classification problems

In the last iteration, ROC-AUC curve and PR-AUC curve will be plotted for users to analyze. For the individual AUC results, users can refer to the dictionary object output. <br>
![ROC Curve that plots True Positive Rate against False Positive Rate](https://github.com/kohjiaxuan/Fraud-Detection-Pipeline/blob/master/ROC_AUC_Curve.PNG)
<br><br>
![PR Curve that plots Precision against Recall](https://github.com/kohjiaxuan/Fraud-Detection-Pipeline/blob/master/PR_AUC_Curve.PNG)

## Variable selection via forward elimination
Updated: 19 May 2020 <br>
Assuming a large amount of training variables, forward selection can be used to prune the number of variables for model training <br>
Refer to <b>forward_elim_binary.py</b> for function to do forward selection and (optional) oversampling <br>
Note that oversampling might cause problems during forward selection due to the training numpy matrix becoming singular and unable to solve the inverse <br>

```
def forward_selection(df, sig_level, response, removelist, sampling='nil', testratio=0):
    """
    :param df: dataframe with both training and response variables
    :param sig_level: significance level to accept/reject var during forward selection
    :param response: name of response var in dataframe
    :param removelist: list of training variables to remove from dataframe
    :param sampling: type of oversampling to use, smote, naive or nil, default: no sampling done
    :param testratio: proportion of dataset to remove out before doing oversampling, default: 0
    :return: list of training variables
    """
```

## Variable selection via backward elimination
Updated: 19 May 2020 <br>
Assuming a large amount of training variables, backward selection can be used to prune the number of variables for model training <br>
Refer to <b>backward_elim_binary.py</b> for function to do backward selection <br>


```
def backward_elimination(x, Y, sl, columns):
    """
    :param x: numpy array of training variables
    :param Y: Y is numpy array of response variable
    :param sl: significance level in float
    :param columns: list of columns in same horizontal order with x
    :return: numpy array of selected x AND list of selected training variables passing sig level
    """
```

