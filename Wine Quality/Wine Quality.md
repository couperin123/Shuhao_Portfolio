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

Check whether there are missing values in the dataset.


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



## Correlation Matrix


```python
# correlation matrix
corrmat = df_red.corr() # correlation matrix calculated from pandas
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f50b792708>




![png](Wine%20Quality_files/Wine%20Quality_9_1.png)



```python
# quality correlation matrix
k = 12 # number of variables for heatmap
cols = corrmat.nlargest(k, 'quality')['quality'].index # select the column names that highly correlates with 'quality'
cm = np.corrcoef(df_red[cols].values.T) # calculate the Pearson product-moment correlation coefficients using numpy
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12,9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 14}, yticklabels=cols.values, xticklabels=cols.values)
```


![png](Wine%20Quality_files/Wine%20Quality_10_0.png)



```python
# scatterplot
sns.set()
cols = ['quality', 'alcohol', 'sulphates', 'citric acid', 'volatile acidity']
sns.pairplot(df_red[cols], size=2.5)
plt.show()
```

    C:\Users\Shuhao\Anaconda3\lib\site-packages\seaborn\axisgrid.py:2071: UserWarning: The `size` parameter has been renamed to `height`; please update your code.
      warnings.warn(msg, UserWarning)
    


![png](Wine%20Quality_files/Wine%20Quality_11_1.png)


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


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score, roc_curve, roc_auc_score, plot_roc_curve
from sklearn.model_selection import cross_val_score, cross_val_predict
```

### Stochastic Gradient Descent Classifier


```python
from sklearn.linear_model import SGDClassifier

# my_model_sgd = SGDClassifier(random_state=42)
my_pipeline_sgd = Pipeline(steps=[('model', SGDClassifier(random_state=42))])
# my_model_sgd.fit(X_train,y_train)
y_scores_sgd = cross_val_predict(my_pipeline_sgd, X_train, y_train, cv=5, method="decision_function")
```


```python
fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_train, y_scores_sgd)
```


```python
auc_score_sgd = roc_auc_score(y_train,y_scores_sgd)
print("Logistic ROC score:",auc_score_sgd)
```

    Logistic ROC score: 0.6872054922764862
    

### Support Vector Machines (SVM)


```python
my_pipeline_svc = Pipeline(steps=[('model', SVC(random_state=42))])
y_scores_svc = cross_val_predict(my_pipeline_svc, X_train, y_train, cv=5, method="decision_function")
```


```python
fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_train, y_scores_svc)
```


```python
auc_score_svc = roc_auc_score(y_train,y_scores_svc)
print("Logistic ROC score:",auc_score_svc)
```

    Logistic ROC score: 0.8110833723409788
    

### Random Forest Classifier


```python
def get_score(n_estimators):
    """Return the average MAE over 5 CV folds of random forest model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    # Replace this body with your own code
    my_pipeline = Pipeline(steps=[('model', RandomForestClassifier(n_estimators=n_estimators, random_state=42))])
    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
                              cv=5,
                              scoring='neg_log_loss')
    return scores.mean()
```


```python
results = {i:get_score(i) for i in range(50,450,50)} # Your code here
```


```python
plt.plot(list(results.keys()), list(results.values()))
plt.show()
```


![png](Wine%20Quality_files/Wine%20Quality_33_0.png)



```python
# n_estimators = 100 has the best result
my_pipeline_rfst = Pipeline(steps=[('model', RandomForestClassifier(n_estimators=100, random_state=42))])
y_probas_rfst = cross_val_predict(my_pipeline_rfst, X_train, y_train, cv=5, method="predict_proba")
```


```python
y_scores_rfst = y_probas_rfst[:, 1] # score = proba of positive class
fpr_rfst, tpr_rfst, thresholds_rfst = roc_curve(y_train, y_scores_rfst)
```


```python
auc_score_rfst = roc_auc_score(y_train,y_scores_rfst)
print("Logistic ROC score:",auc_score_rfst)
```

    Logistic ROC score: 0.8983486763405627
    


```python
plt.plot([0,1],[0,1],linestyle="-",label="Generalized Model Curve")
plt.plot(fpr_rfst, tpr_rfst,"b:", label='Random Forest')
plt.legend()
plt.title(f'Logistic ROC score: {auc_score_rfst:.3f}', fontsize=16)
plt.show()
```


![png](Wine%20Quality_files/Wine%20Quality_37_0.png)


## Comparing ROC Curves across different models


```python
plt.plot([0,1],[0,1],"C1-",label="Generalized Model Curve")
plt.plot(fpr_rfst, tpr_rfst,"C2-", label='Random Forest')
plt.plot(fpr_sgd, tpr_sgd,"C3-", label='SGD')
plt.plot(fpr_svc, tpr_svc,"C4-", label='SVC')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'(RF, SGD) ROC score: ({auc_score_rfst:.3f}, {auc_score_sgd:.3f})', fontsize=16)
plt.show()
```


![png](Wine%20Quality_files/Wine%20Quality_39_0.png)



```python
df_red.dtypes
```




    fixed acidity           float64
    volatile acidity        float64
    citric acid             float64
    residual sugar          float64
    chlorides               float64
    free sulfur dioxide     float64
    total sulfur dioxide    float64
    density                 float64
    pH                      float64
    sulphates               float64
    alcohol                 float64
    quality                   int64
    binary quality            int32
    dtype: object




```python
df_red.head()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>binary quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

jupyter nbconvert Wine_Quality.ipynb  --to markdown
```


      File "<ipython-input-8-1f9deafd051d>", line 2
        nbconvert Wine_Quality.ipynb  --to markdown
                             ^
    SyntaxError: invalid syntax
    



```python

```
