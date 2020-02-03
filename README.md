# Fraud-Detection-Pipeline
A structured data pipeline for classification problems that does multiple purposes like scaling, sampling, k-fold cross validation with evaluation metrics. <br>
It reduces the need for users to rewrite a lot of code as it's reusability is very high.<br>
Refer to sklearn_classification_pipeline.py for the full code <br><br>
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
## To run pipeline, create new class object modelpipeline()
## Next, execute modelpipeline.runmodel(...)
# Parameters are:
## df - DataFrame that has went through data cleaning, processing, feature engineering, containing both features and response
### No standardization/scaling is required as there is built in function for that
## varlist - List of all variables/features to use in model, including the response variable. 
### This allows user to do feature selection first and select only the relevant features before running the model
## response - Name of the response variable in string format
## sampletype - For undersampled data, you can use 'naive', 'smote' or 'adasyn' oversampling. If other strings are input, then no oversampling is done.
## modelname - Choose the type of model to run - user can add more models as required using the if/else clause to check this string input in the buildmodel function
## text - Text title to put for the confusion matrix output in each iteration of the n-folds stratified cross validation
## n-fold - number of folds for the stratified cross validation
