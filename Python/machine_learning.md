# Machine Learning <!-- omit in toc -->

---

## Table of Contents <!-- omit in toc -->

- [Definitions](#definitions)
  - [Bias Variance Trade-Off](#bias-variance-trade-off)
- [PRE-PROCESSING](#pre-processing)
  - [Train Test Split](#train-test-split)
  - [Scalers](#scalers)
  - [Imputers](#imputers)
  - [Cross Validation](#cross-validation)
- [SUPERVISED LEARNING](#supervised-learning)
  - [CLASSIFICATION MODELS](#classification-models)
    - [K-Nearest Neighbors](#k-nearest-neighbors)
    - [Logistic Regression](#logistic-regression)
    - [Support Vector Machines (SVC)](#support-vector-machines-svc)
    - [LinearSVC](#linearsvc)
    - [Random Forests](#random-forests)
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

## Definitions

**Accuracy** - the porportion of correct predictions made by a model over the total # of predictions.

    # of correct predictions (TP + TN) / total # of predictions

**Missclassification (Error) Rate** - the porportion of incorrect predictions made by a model over the total # of predictions.

    # of incorrect predictions (FP + FN) / total # of predictions

**Recall** - the ability of a model to find all the revelant cases within a dataset.

    # of true positives (TP) / [# of true positives (TP) + # of false negatives (FN)]

**Precision** - the porportion of cases the models says was revelant and cases that were actually revelant.

    # of true positives (TP) / [# of true positives (TP) + # of false positives (FP)]

**F1 Score** - the optimal blend of precision and recall to take into account both metrics. F1 provides the harmonic mean, which unlike the simple mean, punishes extreme values.

    F1 = 2 * (precision * recall) / (precision + recall)

**Bootstramp Samples** - sampling from the dataset with replacement

### Bias Variance Trade-Off

- Low Bias / Low Variance - most accurate model
- Low Bias / High Variance - accurate predictions and more dispersed outliers
- High Bias / Low Variance - biased predictions and minimal outliers
- High Bias / High Variance biased predictions and more disperesed outliers

## PRE-PROCESSING

### Train Test Split

Split the data into separate **Train (X_train, y_train)** and **Test (X_test, y_test)** sets

```python
from sklearn.model_selection import train_test_split

# target values
y = df['Target_Column']

# feature values
X = df.drop('Target_Column', axis=1)

# splitting the data into train/test (70%/30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=True)

# test_size = % of test set, stratify = proportionate labels in each set
```

### Scalers

Scale feature data using **scale** in **sklearn.preprocessing**

```python
from sklearn.preprocessing import scale

# feature values
X = df.drop('Target_Column', axis=1)

# scaled feature values
X_scaled = scale(X, axis=0)

# axis=0 standardizes each feature
# axis=1 standardizes each record
```

Scale data using **StandardScaler** transformer in **sklearn.preprocessing**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fitting feature data using scaler
scaler.fit(X)

# each feature to have MEAN=0 and STD=1
StandardScaler(copy=True, with_mean=True, with_std=True)

# transform data for scaled feature values
samples_scaled = scaler.transform(X)
```

### Imputers

Replace missing data using **Imputer** transformer in **sklearn.preprocessing**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

# feature values
X = df.drop('Target_Column', axis=1)

# axis=0 indicates columns, axis=1 indicates records
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# filling missing data using imputer
imp.fit(X)

# transform data for imputed feature values
X = imp.transform(X)
```

### Cross Validation

Using a **Linear Regression** model:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# target values
y = df['Target_Column']

# feature values
X = df.drop('Target_Column', axis=1)

reg = LinearRegression()

# CROSS VALIDATING the model; cv= # of CROSS-VALIDATION folds
cv_results = cross_val_score(reg, X, y, cv=5)

# cv_results = array of CV scores (R-squared for LR)
```

## SUPERVISED LEARNING

### CLASSIFICATION MODELS

- Accuracy

  - great with balanced classes, but not great with unbalanced classes (ie. 100 images with 99 dog images, 1 cat image)

- Precision
  - measured by the ability of the model to identify only the revelant data points.

#### K-Nearest Neighbors

- Classification algorithm that operates as:

  - calculates the distance from "x" to all points in the data
  - sorts the points by increasing distance from "x"
  - predicts the majority label of the "k" closest points

- Does not handle categorical data well

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

y = df['Target_Column']
X = df.drop('Target_Column', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# defining n_neighbors for model
knn = KNeighborsClassifier(n_neighbors=6)

# fit the model to training data
knn.fit(X_train, y_train)

# predict the labels for the test data
pred = knn.predict(X_test)

# scoring the model accuracy with the train/test data
train_score = knn.score(X_train, y_train)
test_score = knn.score( X_test, y_test)
```

#### Logistic Regression

- Logistic (Sigmoid) Function takes in any value and outputs it between 0 and 1.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

y = df['Target_Column']
X = df.drop('Target_Column', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# instantiating a LOGISTIC_REGRESSION classifier
lr = LogisticRegression()

# fit the model to the training data
lr.fit(X_train, y_train)

# predict the labels for the test data
y_pred = lr.predict(X_test)

# scoring the model accuracy with the train/test data
train_score = lr.score(X_train, y_train)
test_score = lr.score(X_test, y_test)
```

#### Support Vector Machines (SVC)

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

y = df['Target_Column']
X = df.drop('Target_Column', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

svm = SVC()

# fit the model to the training data
svm.fit(X_train, y_train)

# predict the labels for the test data
pred = svm.predict(X_test)

# scoring the model accuracy with the train/test data
train_score = svm.score(X_train, y_train)
test_score = svm.score(X_test, y_test)
```

#### LinearSVC

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

y = df['Target_Column']
X = df.drop('Target_Column', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

svm = LinearSVC()

# fit the model to the training data
svm.fit(X_train, y_train)

# predict the labels for the test data
pred = svm.predict(X_test)

# scoring the model accuracy with the train/test data
train_score = svm.score(X_train, y_train)
test_score = svm.score(X_test, y_test)
```

#### Random Forests

Uses many trees with a random sample of features chosen as the split. Alleviates one very strong feature from creating an ensemble of highly similar trees that use the strong feature at the top split.

- a new random sample of features is chosen for every single tree, at every single split.
- in a classification random forest model, `m` features is typically chosen to be the `sqrt(p)` or square root of the `p` full set of features.

### REGRESSION MODELS

- Mean Absolute Error (MAE)

  - the mean of the absolute value of errors `ie. true value - predicted value`, easy to understand but won't punish large errors.

- Mean Squared Error (MSE)

  - the mean of the value of errors squared `ie. (true value - predicted_value)^2`, accounts for large errors but also squares units.

- Root Mean Square Error (RMSE)
  - the square root of the mean of the values of errors squared `ie. sqrt((true_value - predicted value)^2)` to punish large values and provides same units as y.

#### Linear Regression

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

y = df['Target_Column']
X = df.drop('Target_Column', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

reg = LinearRegression()

# fit the model to the training data
reg.fit(X_train, y_train)

# predict the labels for the test data
y_pred = reg.predict(X_test)

# scoring the model accuracy with the train/test data
train_score = reg.score(X_train, y_train)
test_score = reg.score(X_test, y_test)
```

#### Ridge

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

y = df['Target_Column']
X = df.drop('Target_Column', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# normalize=True normalizes all variables
ridge = Ridge(alpha=0.1, normalize=True)

# fit the model to the training data
ridge.fit(X_train, y_train)

# predict the labels for the test data
ridge_pred = ridge.predict(X_test)

# scoring the model accuracy with the train/test data
train_score = ridge.score(X_train, y_train)
test_score = ridge.score(X_test, y_test)
```

#### Lasso

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

y = df['Target_Column']
X = df.drop('Target_Column', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# normalize=True normalizes all variables
lasso = Lasso(alpha=0.1, normalize=True)

# fit the model to the training data
lasso.fit(X_train, y_train)

# predict the labels for the test data
lasso_pred = lasso.predict(X_test)

# scoring the model accuracy with the train/test data
train_score = lasso.score(X_train, y_train)
test_score = lasso.score(X_test, y_test)
```

### FEATURE TUNING

#### Confusion Matrix

<p align="center">
<img src="confusion-matrix.png" width="600" height="400">
</p>

Fundamentally a Confusion Matrix provides ways to compare predicted values versus true values.

- **True Positive (TP)** - model predicts positive, outcome is positive
- **False Positive (FP) Type I Error** - model predicts positive, outcome is negative
- **True Negative (TN)** - model predicts negative, outcome is negative
- **False Negative (FN) Type II Error** - model predicts negative, outcome is positive

Using a **K-Nearest Neighbors** Model:

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

y = df['Target_Column']
X = df.drop('Target_Column', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# defining n_neighbors for the model
knn = KNeighborsClassifier(n_neighbors=6)

# fit the model to the training data
knn.fit(X_train, y_train)

# predict the labels for the test data
y_pred = knn.predict(X_test)

# generate the confusion_matrix on actual/predicted labels
confusion_m = confusion_matrix(y_test, y_pred)

# print confusion_m = [[TP, FN], [FP,TN]]
print(confusion_m)
```

#### Classification Report

Using a **K-Nearest Neighbors** Model:

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

y = df['Target_Column']
X = df.drop('Target_Column', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# defining n_neighbors for the model
knn = KNeighborsClassifier(n_neighbors=6)

# fit the model to the training data
knn.fit(X_train, y_train)

# predict the labels for the test data
y_pred = knn.predict(X_test)

# generate the classification report on actual/predicted labels
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

y = df['Target_Column']
X = df.drop('Target_Column', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

lr = LogisticRegression()

# predict_proba gives probabilities w/out threshold; [:,1] are labels for the '1' binary label
y_pred_prob = lr.predict_proba(X_test)[:,1]

# unpack ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

'''
fpr = FALSE POSITIVE RATE
tpr = TRUE POSITIVE RATE
'''
```

#### AUC (Area under ROC Curve) - Large AUC = better model

Produces one AUC Score.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

y = df['Target_Column']
X = df.drop('Target_Column', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

lr = LogisticRegression()

# fit the model to the training data
lr.fit(X_train, y_train)

# predict_proba gives probabilities w/out threshold; [:,1] are labels for the '1' binary label
y_pred_prob = lr.predict_proba(X_test)[:,1]

# compute the ROC_AUC_SCORE
score = roc_auc_score(y_test, y_pred_prob)
```

Using **cross_val_score** from **sklearn.model_selection** (Cross Validation); produces multiple AUC Scores

```python
from sklearn.model_selection import cross_val_score


y = df['Target_Column']
X = df.drop('Target_Column', axis=1)

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
