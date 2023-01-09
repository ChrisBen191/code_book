# Machine Learning <!-- omit in toc -->

---

## Table of Contents <!-- omit in toc -->

- [PRE-PROCESSING](#pre-processing)
  - [Cross Validation](#cross-validation)
- [SUPERVISED LEARNING](#supervised-learning)
  - [CLASSIFICATION MODELS](#classification-models)
    - [K-Nearest Neighbors](#k-nearest-neighbors)
    - [Logistic Regression](#logistic-regression)
    - [Support Vector Machines (SVC)](#support-vector-machines-svc)
    - [LinearSVC](#linearsvc)
  - [REGRESSION MODELS](#regression-models)
    - [Linear Regression](#linear-regression)
    - [Ridge](#ridge)
    - [Lasso](#lasso)
  - [FEATURE TUNING](#feature-tuning)
    - [Confusion Matrix](#confusion-matrix)
    - [Classification Report](#classification-report)
    - [ROC (Receiver Operating Characteristic) Curve](#roc-receiver-operating-characteristic-curve)
    - [AUC (Area under ROC Curve) - Large AUC = better model](#auc-area-under-roc-curve---large-auc--better-model)
  - [HYPERPARAMETER TUNING](#hyperparameter-tuning)
  - [GridSearchCV (Grid Search Cross Validation)](#gridsearchcv-grid-search-cross-validation)
    - [RandomizedSearchCV (Randomized GridSearchCV)](#randomizedsearchcv-randomized-gridsearchcv)
  - [PIPELINES](#pipelines)

## PRE-PROCESSING

Scale data using **scale** in **sklearn.preprocessing**

```python
# importing SCALE
from sklearn.preprocessing import scale

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# axis=0 standardizes each feature, axis=1 standardizes each record
X_scaled = scale(X, axis=0)
```

Scale data using **StandardScaler** transformer in **sklearn.preprocessing**

```python
# importing STANDARD_SCALER
from sklearn.preprocessing import StandardScaler

# initializing the STANDARD SCALER
scaler = StandardScaler()

# fitting scaler to the data being scaled
scaler.fit(X)

# each feature to have MEAN=0 and STD=1
StandardScaler(copy=True, with_mean=True, with_std=True)

# transform data
samples_scaled = scaler.transform(X)
```

Replace missing data using **Imputer** transformer in **sklearn.preprocessing**

```python
# importing TRAIN_TEST_SPLIT and IMPUTER
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# instantiating IMPUTER transformer, axis=0 indicates columns
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# fitting IMPUTER to the data
imp.fit(X)

# transforming the X data
X = imp.transform(X)
```

Splits the data into separate **Train (X_train, y_train)** and **Test (X_test, y_test)** sets

```python
# importing TRAIN_TEST_SPLIT
from sklearn.model_selection import train_test_split

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=True)

# test_size = % of test set, stratify = porportionate labels in each set
```

### Cross Validation

Using a **Linear Regression** model:

```python
# importing CROSS_VALIDATION and LINEAR_REGRESSION
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# instantiating a LINEAR_REGRESSION regressor
reg = LinearRegression()

# CROSS VALIDATING the model; cv= # of CROSS-VALIDATION folds
cv_results = cross_val_score(reg, X, y, cv=5)

# cv_results = array of CV scores (R-squared for LR)
```

## SUPERVISED LEARNING

### CLASSIFICATION MODELS

#### K-Nearest Neighbors

```python
# importing TRAIN_TEST_SPLIT and KNEIGHBORS_CLASSIFIER
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# instantiating a KNEIGHBORS classifier
knn = KNeighborsClassifier(n_neighbors=6)

# fit the model to the training data
knn.fit(X_train, y_train)

# predict the labels for the TEST data
pred = knn.predict(X_test)

# scoring the model accuracy with the TRAIN and TEST data
train_score = knn.score(X_train, y_train)
test_score = knn.score( X_test, y_test)
```

#### Logistic Regression

```python
# importing TRAIN_TEST_SPLIT and LOGISTIC_REGRESSION
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# instantiating a LOGISTIC_REGRESSION classifier
lr = LogisticRegression()

# fit the model to the training data
lr.fit(X_train, y_train)

# predict the labels for the TEST data
y_pred = lr.predict(X_test)

# scoring the model accuracy with the TRAIN and TEST data
train_score = lr.score(X_train, y_train)
test_score = lr.score(X_test, y_test)
```

#### Support Vector Machines (SVC)

```python
# importing TRAIN_TEST_SPLIT and SVC
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# instantiating a SVC classifier
svm = SVC()

# fit the model to the training data
svm.fit(X_train, y_train)

# predict the labels for the TEST data
pred = svm.predict(X_test)

# scoring the model accuracy with the TRAIN and TEST data
train_score = svm.score(X_train, y_train)
test_score = svm.score(X_test, y_test)
```

#### LinearSVC

```python
# importing TRAIN_TEST_SPLIT and LINEARSVC
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# instantiating a LINEARSVC classifier
svm = LinearSVC()

# fit the model to the training data
svm.fit(X_train, y_train)

# predict the labels for the TEST data
pred = svm.predict(X_test)

# scoring the model accuracy with the TRAIN and TEST data
train_score = svm.score(X_train, y_train)
test_score = svm.score(X_test, y_test)
```

### REGRESSION MODELS

#### Linear Regression

```python
# importing TRAIN_TEST_SPLIT and LINEAR_REGRESSION
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# # instantiating a LINEAR_REGRESSION regressor
reg = LinearRegression()

# fit the model to the training data
regl.fit(X_train, y_train)

# predicting the labels for the X_TEST set
y_pred = reg.predict(X_test)

# scoring the model accuracy with the TRAIN and TEST data
train_score = reg.score(X_train, y_train)
test_score = reg.score(X_test, y_test)
```

#### Ridge

```python
# importing TRAIN_TEST_SPLIT and RIDGE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# instantiating a RIDGE regressor; normalize=True normalizes all variables
ridge = Ridge(alpha=0.1, normalize=True)

# fit the model to the training data
ridge.fit(X_train, y_train)

# predicting the labels for the X_TEST set
ridge_pred = ridge.predict(X_test)

# scoring the model accuracy with the TRAIN and TEST data
train_score = ridge.score(X_train, y_train)
test_score = ridge.score(X_test, y_test)
```

#### Lasso

```python
# importing TRAIN_TEST_SPLIT and LASSO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# instantiating a LASSO regressor; normalize=True normalizes all variables
lasso = Lasso(alpha=0.1, normalize=True)

# fit the model to the training data
lasso.fit(X_train, y_train)

# predicting the labels for the X_TEST set
lasso_pred = lasso.predict(X_test)

# scoring the model accuracy with the TRAIN and TEST data
train_score = lasso.score(X_train, y_train)
test_score = lasso.score(X_test, y_test)
```

### FEATURE TUNING

#### Confusion Matrix

Using a **K-Nearest Neighbors** Model:

```python
# importing TRAIN_TEST_SPLIT, KNEIGHBORS_CLASSIFIER, and CONFUSION_MATRIX
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# instantiating a KNEIGHBORS_CLASSIFIER
knn = KNeighborsClassifier(n_neighbors=6)

# fit the model to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# generate CONFUSION_MATRIX on actual/predicted labels
confusion_m = confusion_matrix(y_test, y_pred)

# print confusion_m = [[TP, FN], [FP,TN]]
print(confusion_m)
```

#### Classification Report

Using a **K-Nearest Neighbors** Model:

```python
# importing TRAIN_TEST_SPLIT, KNEIGHBORS_CLASSIFIER, and CLASSIFICATION_REPORT
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# instantiating a KNEIGHBORS_CLASSIFIER
knn = KNeighborsClassifier(n_neighbors=6)

# fit the model to the training data
knn.fit(X_train, y_train)

# predicting on the TEST set
y_pred = knn.predict(X_test)

# generate CLASSIFICATION_REPORT on actual/predicted labels
classification_r = classification_report(y_test, y_pred)

# SUPPORT column provides # of samples of the TRUE RESPONSE that lie in that class
print(classification_report(y_test, y_pred))

'''
SPAM E-MAIL EXAMPLE:
HIGH PRECISION = not many emails predicted as spam
HIGH RECALL = predicted most spam emails correctly
'''
```

#### ROC (Receiver Operating Characteristic) Curve

```python
# importing TRAIN_TEST_SPLIT and ROC_CURVE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# instantiating a LOGISTIC_REGRESSION classifier
lr = LogisticRegression()

# PREDICT_PROBA predicts probabilities w/out threshold; [:,1] are labels for the '1' binary label
y_pred_prob = lr.predict_proba(X_test)[:,1]

# unpack ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

'''
fpr = FALSE POSITIVE RATE
tpr = TRUE POSITIVE RATE
'''
```

#### AUC (Area under ROC Curve) - Large AUC = better model

Using **roc_auc_score** in **sklearn.metrics**; produces one AUC Score.

```python
# importing TRAIN_TEST_SPLIT and ROC_AUC_SCORE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# instantiating a LOGISTIC_REGRESSION classifier
lr = LogisticRegression()

# fit the LOG_REG model
lr.fit(X_train, y_train)

# compute predicted probabilities of model labels w/out threshold;
# [:,1] are labels for the '1' binary label
y_pred_prob = lr.predict_proba(X_test)[:,1]

# compute the ROC_AUC_SCORE
score = roc_auc_score(y_test, y_pred_prob)
```

Using **cross_val_score** from **sklearn.model_selection** (Cross Validation); produces multiple AUC Scores

```python
# importing CROSS_VAL_SCORE
from sklearn.model_selection import cross_val_score

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# instantiating a LOGISTIC_REGRESSION classifier
lr = LogisticRegression()

# CROSS_VALIDATING the model; scoring='roc_auc' provides AUC (Area Under ROC Curve) Scores
cv_results = cross_val_score(lr, X, y, cv=5, scoring='roc_auc')

# compute the average ROC_AUC_SCORE
mean_score = np.mean(cv_results)
```

### HYPERPARAMETER TUNING

### GridSearchCV (Grid Search Cross Validation)

Using a **Logistic Regression** Model

```python
# importing TRAIN_TEST_SPLIT, LOGISTIC_REGRESSION, and GRIDSEARCH_CV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# setup HYPERPARAMETER_GRID for model hyperparameter(s)
param_grid = {'C': np.arange(0, 10, 1)}

# instantiating a LOGISTIC_REGRESSION classifier
lr = LogisticRegression()

# intantiating the GRIDSEARCH_OBJECT
lr_cv = GridSearchCV(lr, param_grid, cv=5)

# fit the LOG_REG model
lr_cv.fit(X_train, y_train)

# returns a DICT of the BEST PREFORMING parameters and values
best_p = lr_cv.best_params_

# returns the SCORE of the BEST PREFORMING parameter
best_score = lr_cv.best_score_
```

Using a **K-Nearest Neighbors** Model

```python
# importing TRAIN_TEST_SPLIT, LOGISTIC_REGRESSION, and GRIDSEARCH_CV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# setup HYPERPARAMETER_GRID for model hyperparameter(s)
param_grid = {'n_neighbors': np.arange(1, 50)}

# instantiating a K_NEAREST classifier
knn = KNeighborsClassifier()

# intantiating the GRIDSEARCH_OBJECT
knn_cv = GridSearchCV(knn, param_grid, cv=5)

# fit the K_NEAREST model
knn_cv.fit(X_train, y_train)

# returns a DICT of the BEST PREFORMING parameters and values
best_p = knn_cv.best_params_

# returns the SCORE of the BEST PREFORMING parameter
best_score = knn_cv.best_score_
```

Using a **ElasticNet** Model

```python
# importing TRAIN_TEST_SPLIT, ELASTIC_NET, MEAN_SQUARED_E, and GRIDSEARCH_CV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# setup HYPERPARAMETER_GRID for model hyperparameter(s)
param_grid = {'l1_ratio': np.linspace(0, 1, 30)}

# instantiating a ELASTIC_NET classifier
elastic_net = ElasticNet()

# intantiating the GRIDSEARCH_OBJECT
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# fit the ELASTIC_NET model
gm_cv.fit(X_train, y_train)

# predicting on the TEST set
y_pred = gm_cv.predict(X_test)

# scoring the ELASTIC_NET model (REGRESSOR)
r2 = gm_cv.score(X_test, y_test)

# computing the MEAN_SQUARED_ERROR using actual/predicted labels
mse = mean_squared_error(y_test, y_pred)

# returns a DICT of the BEST PREFORMING parameters and values
best_p = gm_cv.best_params_

# returns the SCORE of the BEST PREFORMING parameter
best_score = gm_cv.best_score_

print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))
```

#### RandomizedSearchCV (Randomized GridSearchCV)

Not as computationally expensive as **GridSearchCV**

```python
# importing TRAIN_TEST_SPLIT, DECISION_TREE, RANDOMIZED_GRIDSEARCH, and RANDINT
from sklearn.model_selection import train_test_split
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# setup HYPERPARAMETER_GRID for model hyperparameter(s)
param_dist = {"max_depth": [3, None], "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9), "criterion": ["gini", "entropy"]}

# instantiating a DECISION_TREE classifier
tree = DecisionTreeClassifier()

# intantiating the GRIDSEARCH_OBJECT
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# fit the DECISION_TREE model
tree_cv.fit(X_train, y_train)

# returns a DICT of the BEST PREFORMING parameters and values
best_p = tree_cv.best_params_

# returns the SCORE of the BEST PREFORMING parameter
best_score = tree_cv.best_score_
```

### PIPELINES

Using **K-Nearest Neighbors** Model; scaling data using **StandardScaler** and comparing scaled/un-scaled scores.

```python

# importing TRAIN_TEST_SPLIT, STANDARD_SCALER, KNEIGHBORS, and PIPELINE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# setup STEPS with TRANSFORMERS/ESTIMATORS
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]

# instantiating PIPELINE
pipeline = Pipeline(steps)

# fit the PIPELINE, computes 'scaler' then 'knn'
knn_scaled = pipeline.fit(X_train, y_train)

# instantiating a KNEIGHBORS classifier, fitting to UN-SCALED data for comparison
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# scoring the model accuracy with the SCALED and UN-SCALED TEST data
scaled_score = knn_scaled.score(X_test, y_test)
unscaled_score = knn_unscaled.score(X_test, y_test)
```

Using **SVC (Support Vector Machine)** Model; producing **Classification Report** on predictions after cleaning up missing data

```python
# importing TRAIN_TEST_SPLIT, SVC, IMPUTER, CLASS_REPORT, and PIPELINE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# creates an array for the TARGET values
y = df['Target_Column'].values

# creates a column for the FEATURES
X = df.drop('Target_Column', axis=1).values

# splitting the data into TRAIN/TEST (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# instantiating IMPUTER transformer, axis=0 indicates columns, strategy = replacement method
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# setup STEPS with TRANSFORMERS/ESTIMATORS
steps = [('imputation', imp), ('SVM', SVC())]

# instantiating PIPELINE
pipeline = Pipeline(steps)

# fit the PIPELINE, computes 'imputation' then 'SVM'
pipeline.fit(X_train, y_train)

# predicting on the TEST set
y_pred = pipeline.predict(X_test)

# generate CLASSIFICATION_REPORT on actual/predicted labels
classification_r = classification_report(y_test, y_pred)

# SUPPORT column provides # of samples of the TRUE RESPONSE that lie in that class
print(classification_report(y_test, y_pred))
```
