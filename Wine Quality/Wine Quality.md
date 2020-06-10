# Wine Quality Predictions

## Problem Definition

Is it possible to predict human wine taste preference based on physicochemical properties of the wine (such as pH, amount of sulfates and alcohol)? The purpose of this project is to answer this question with different machine learning methods.

## Retrieve the Dataset

The two datasets correspond to the red and white "Vinho Verde" wine (from Northern Portugal). For more detail please consult the original journal article. [[Cortez et al., 2009]](https://doi.org/10.1016/j.dss.2009.05.016)
This dataset is available from the UCI machine learning repository: https://archive.ics.uci.edu/ml/datasets/wine+quality

## Attribute Information

Predictor variables (all numerical):

1. fixed acidity           
2. volatile acidity        
3. citric acid             
4. residual sugar          
5. chlorides               
6. free sulfur dioxide     
7. total sulfur dioxide    
8. density                 
9. pH                      
10. sulphates               
11. alcohol                 

Target variable (categorical):

12. quality (0-10 scale)       

## Binary Classification

To simply the problem, we will only consider the red wines and classify these wines as "GOOD" and "BAD" ones by setting the bar at quality 6.5.


```python
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
```


```python
df_red = pd.read_csv('./input/winequality-red.csv',sep=';')
```

Check whether there are missing values in the dataset. Fortunately this dataset is clean so there is none.


```python
df_red.isnull().sum()
```




    fixed acidity           0
    volatile acidity        0
    citric acid             0
    residual sugar          0
    chlorides               0
    free sulfur dioxide     0
    total sulfur dioxide    0
    density                 0
    pH                      0
    sulphates               0
    alcohol                 0
    quality                 0
    dtype: int64



## Exploratory Data Analysis

## Correlation Matrix


```python
# correlation matrix
corrmat = df_red.corr() # correlation matrix calculated from pandas
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b7345f4c88>




![png](Wine%20Quality_files/Wine%20Quality_9_1.png)



```python
# quality correlation matrix
k = 12 # number of variables for heatmap
cols = corrmat.nlargest(k, 'quality')['quality'].index # select the column names that highly correlates with 'quality'
cm = np.corrcoef(df_red[cols].values.T) # calculate the Pearson product-moment correlation coefficients using numpy
cmmask = np.triu(cm).astype(bool) # Mask the upper triangle to show only the lower triangle of the correlation matrix
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12,9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 14}, 
                 mask=cmmask,
                 yticklabels=cols.values, xticklabels=cols.values)
```


![png](Wine%20Quality_files/Wine%20Quality_10_0.png)



```python
# scatterplot
sns.set()
cols = ['quality', 'alcohol', 'sulphates', 'citric acid', 'volatile acidity']
sns.pairplot(df_red[cols], hue = 'quality')
plt.show()
```


![png](Wine%20Quality_files/Wine%20Quality_11_0.png)


Add a columm 'binary quality' with values 0 and 1 that corresponds to 'quality' values lower or higher than 6.5.


```python
df_red['binary quality'] = np.where(df_red.quality < 6.5, 0, 1)
```


```python
# count the number of GOOD/BAD wine
df_red['binary quality'].value_counts()
```




    0    1382
    1     217
    Name: binary quality, dtype: int64




```python
X = df_red.loc[:, ~df_red.columns.isin(['quality','binary quality'])]
y = df_red['binary quality']
```

## Stratified sampling


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
```


```python
y_train.value_counts()
```




    0    1105
    1     174
    Name: binary quality, dtype: int64




```python
y_test.value_counts()
```




    0    277
    1     43
    Name: binary quality, dtype: int64



## Machine Learning Models

The selected metric to compare between machine learning methods is the area under the receiver operating characteristic (ROC) curve, also called the **ROC AUC score**. Calculating this score requires generating the ROC curve with each machine learning method. 

The ROC curve can be computed by the `roc_curve()` function from the `sklearn.metrics` module. The `roc_curve()` function requires two input arguments `y_true` and `y_score`, where `y_true` represents the true binary labels and `y_score` represents the target scores of all instances, which can be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as returned by "decision_function" on some classifiers). 


```python
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score, roc_curve, roc_auc_score, plot_roc_curve, precision_recall_curve
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
```

### Logistic Regression

Notice that L2 regulaization is applied by default in the Logistic Regression function of Scikit-Learn. 


```python
my_pipeline_logreg = Pipeline(steps=[('model',LogisticRegression(random_state=42, max_iter=2000))])
```

Use grid search to explore optimal value of inverse of regularization strength $C$ (must be a positive float). Keep in mind that the default of GridSearchCV method is to use **5-fold** cross validation.


```python
param_grid = [{'model__penalty': ['l2'], 'model__C': list(range(10,101,10))}, 
              {'model__penalty': ['none']}]
grid_logreg = GridSearchCV(my_pipeline_logreg, param_grid)

%time grid_logreg.fit(X_train, y_train)
print(grid_logreg.best_params_)
```

    Wall time: 11.3 s
    {'model__C': 40, 'model__penalty': 'l2'}
    


```python
y_probas_logreg = cross_val_predict(grid_logreg.best_estimator_, X_train, y_train, cv=5, method="predict_proba")
```


```python
y_scores_logreg = y_probas_logreg[:, 1] # score = proba of positive class
fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_train, y_scores_logreg)
```


```python
auc_score_logreg = roc_auc_score(y_train,y_scores_logreg)
print("ROC AUC score:",auc_score_logreg)
```

    ROC AUC score: 0.8638009049773756
    

### Linear Support Vector Machine with Stochastic Gradient Descent


```python
# SVMs are sensitive to the feature scales, so feature scaling is applied
my_pipeline_sgd = Pipeline(steps=[('scaler', StandardScaler()),
                                  ('model', SGDClassifier(random_state=42))])
```

Use grid search to explore optimal value of hyperparameter $\alpha$. 


```python
param_grid = {'model__alpha': [1e-4,1e-3,1e-2,1e-1,1]}
grid_sgd = GridSearchCV(my_pipeline_sgd, param_grid)

%time grid_sgd.fit(X_train, y_train)
print(grid_sgd.best_params_)
```

    Wall time: 212 ms
    {'model__alpha': 0.01}
    


```python
y_scores_sgd = cross_val_predict(grid_sgd.best_estimator_, X_train, y_train, cv=5, method="decision_function")
# plt.hist(y_scores_sgd)
```


```python
fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_train, y_scores_sgd)
```


```python
auc_score_sgd = roc_auc_score(y_train,y_scores_sgd)
print("ROC AUC score:",auc_score_sgd)
```

    ROC AUC score: 0.8199823165340406
    

### Support Vector Machines (SVM)


```python
# SVMs are sensitive to the feature scales, so feature scaling is applied
# RBF kernel performs better than linear kernel in this case 
my_pipeline_svc = Pipeline(steps=[('scaler', StandardScaler()), 
                                  ('model', SVC(random_state=42, kernel="rbf"))])
```

Use grid search to explore combinations of hyperparameters: the margin hardness $C$ and the radial basis function parameter $\gamma$.


```python
param_grid = {'model__C': [5, 10, 20, 40],
              'model__gamma': [1e-3,1e-2,1e-1,1,10]}
grid_svc = GridSearchCV(my_pipeline_svc, param_grid)

%time grid_svc.fit(X_train, y_train)
print(grid_svc.best_params_)
```

    Wall time: 5.17 s
    {'model__C': 10, 'model__gamma': 1}
    


```python
y_scores_svc = cross_val_predict(grid_svc.best_estimator_, X_train, y_train, cv=5, method="decision_function")
# plt.hist(y_scores_svc)
```


```python
fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_train, y_scores_svc)
```


```python
auc_score_svc = roc_auc_score(y_train,y_scores_svc)
print("ROC AUC score:",auc_score_svc)
```

    ROC AUC score: 0.8697144640349508
    

### Decision Tree Classfifier


```python
my_pipeline_dtc = Pipeline(steps=[('model', DecisionTreeClassifier(random_state=42))])
```

Use grid search to find good hypperparameter value for `max_leaf_nodes`.


```python
param_grid = {'model__max_leaf_nodes': list(range(2, 100))}
grid_dtc = GridSearchCV(my_pipeline_dtc, param_grid)

%time grid_dtc.fit(X_train, y_train)
print(grid_dtc.best_params_)
```

    Wall time: 3.95 s
    {'model__max_leaf_nodes': 25}
    


```python
y_probas_dtc = cross_val_predict(grid_dtc.best_estimator_, X_train, y_train, cv=5, method="predict_proba")
```


```python
y_scores_dtc = y_probas_dtc[:, 1] # score = proba of positive class
fpr_dtc, tpr_dtc, thresholds_dtc = roc_curve(y_train, y_scores_dtc)
```


```python
auc_score_dtc = roc_auc_score(y_train,y_scores_dtc)
print("ROC AUC score:",auc_score_dtc)
```

    ROC AUC score: 0.7680995475113123
    

### Random Forest Classifier


```python
# n_estimators = 100 has the best result
my_pipeline_rfst = Pipeline(steps=[('model', RandomForestClassifier(random_state=42))])
```

Use grid search to find good hypperparameter values for `max_leaf_nodes` and `n_estimators` (which controls the number of trees in the forest).


```python
param_grid = {'model__n_estimators': list(range(20, 121, 20)),
             'model__max_leaf_nodes': list(range(80, 121, 10))}
grid_rfst = GridSearchCV(my_pipeline_rfst, param_grid, n_jobs=-1, verbose=2)

%time grid_rfst.fit(X_train, y_train)
print(grid_rfst.best_params_)
```

    Fitting 5 folds for each of 30 candidates, totalling 150 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:    4.1s
    [Parallel(n_jobs=-1)]: Done 150 out of 150 | elapsed:   11.0s finished
    

    Wall time: 11.3 s
    {'model__max_leaf_nodes': 100, 'model__n_estimators': 60}
    

Output each feature's importance from the best model:


```python
for name, score in zip(list(X_train.columns),grid_rfst.best_estimator_.named_steps['model'].feature_importances_):
    print(f'{name}: {score:.4f}')

```

    fixed acidity: 0.0707
    volatile acidity: 0.1037
    citric acid: 0.0918
    residual sugar: 0.0698
    chlorides: 0.0764
    free sulfur dioxide: 0.0603
    total sulfur dioxide: 0.0822
    density: 0.0985
    pH: 0.0603
    sulphates: 0.1184
    alcohol: 0.1679
    


```python
y_probas_rfst = cross_val_predict(grid_rfst.best_estimator_, X_train, y_train, cv=5, method="predict_proba")
```


```python
y_scores_rfst = y_probas_rfst[:, 1] # score = proba of positive class
fpr_rfst, tpr_rfst, thresholds_rfst = roc_curve(y_train, y_scores_rfst)
```


```python
auc_score_rfst = roc_auc_score(y_train,y_scores_rfst)
print("ROC AUC score:",auc_score_rfst)
```

    ROC AUC score: 0.9064154574296562
    

### Plot the ROC Curve

Here the ROC curve is plotted with the threshold values of probability (if the predicted probability of the positive class is larger than this threshold, then it will be classfied as positive). In the upper right corner of the ROC Curve, this threshold is 0 and hence every training instance will be classified as 'Good wine', where at the lower left corner, all the training instances will be classifed as 'Not so good wine'.


```python
plt.plot([0,1],[0,1],linestyle="-",label="Generalized Model Curve")
plt.plot(fpr_rfst, tpr_rfst,"k-", label='Random Forest')
sc = plt.scatter(fpr_rfst, tpr_rfst, c=thresholds_rfst, cmap='cool')
sc.set_clim(0,1)
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel('Threshold Probability')
plt.legend()
plt.title(f'ROC AUC score: {auc_score_rfst:.3f}', fontsize=16)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
```


![png](Wine%20Quality_files/Wine%20Quality_60_0.png)


## Comparing ROC Curves across different models


```python
plt.plot([0,1],[0,1],"C1-",label="Generalized Model Curve")
plt.plot(fpr_rfst, tpr_rfst,"C2-", label='Random Forest')
plt.plot(fpr_sgd, tpr_sgd,"C3-", label='SGD')
plt.plot(fpr_svc, tpr_svc,"C4-", label='SVC')
plt.plot(fpr_logreg, tpr_logreg,"C5-", label='Logistic Regression')
plt.plot(fpr_dtc, tpr_dtc,"C6-", label='Decision Tree')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title(f'(RF, SGD) ROC score: ({auc_score_rfst:.3f}, {auc_score_sgd:.3f})', fontsize=16)
plt.show()
```


![png](Wine%20Quality_files/Wine%20Quality_62_0.png)



```python
models = pd.DataFrame({
    'Model': ['SVM (RBF)', 
              'Linear SVC', 
              'Logistic Regression', 
              'Random Forest',
              'Decision Tree'],
    'ROC AUC Score': [auc_score_svc,auc_score_sgd,auc_score_logreg,
              auc_score_rfst, auc_score_dtc]})
models.sort_values(by='ROC AUC Score', ascending=False).set_index('Model')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ROC AUC Score</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Random Forest</th>
      <td>0.906415</td>
    </tr>
    <tr>
      <th>SVM (RBF)</th>
      <td>0.869714</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.863801</td>
    </tr>
    <tr>
      <th>Linear SVC</th>
      <td>0.819982</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.768100</td>
    </tr>
  </tbody>
</table>
</div>



## Final Model: Voting classifier


```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[('logreg', grid_logreg.best_estimator_),
                ('svc', grid_svc.best_estimator_),
                ('rfst', grid_rfst.best_estimator_)],
    voting='hard')
```


```python
for clf in (grid_logreg.best_estimator_,
            grid_svc.best_estimator_,
            grid_rfst.best_estimator_,
            voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    if clf == voting_clf:
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    else:
        print(clf.named_steps['model'].__class__.__name__, accuracy_score(y_test, y_pred))
```

    LogisticRegression 0.89375
    SVC 0.921875
    RandomForestClassifier 0.934375
    VotingClassifier 0.9375
    

## Conclusion

In this project, we used physicochemical properties of wine to build a wien quality predictor. Five different classifiers (Logistic Regression, SVM with radial basis function kernel, linear SVM, Decision Tree, and Random Forest) were tested, and the individual classifier built by Random Forest has the best performance. The performance can be further enhanced by creating a voting classifier among the best individual classifier (Random Forest, SVM, and Logistic Regression).

## Future Work

It will be interesting to investigate whether boosting (like adaptive boosting and gradient boosting) and stacking ensemble methods can further imporve the performance.
