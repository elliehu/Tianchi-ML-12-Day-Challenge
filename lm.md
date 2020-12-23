# Logit Model

## **Part 1 Intro to LM**

 -  pros
>   - easy to implement
>   - high speed with less requirement for spacing

 - cons
>  - tend to underfit
>  - low accuracy, baseline model

 - applications
> - GBDT(Gradient Boosting Dicision Tree) + LM --> fraud transcation detection of credit card
> - CTR(click through rate) estimation
  
## **Part 2 Coding**

1. packages preparation
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
```

2. model training
```python
x_features = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
y_lable = np.array([0,0,0,1,1,1])

lr_clf = LogisticRegression()

lr_clf = lr_clf.fit(x_feautures, y_lable) #其拟合方程为 y=w0+w1*x1+w2*x2
```

3. model parameters
```python
print('weight of LM:', lr_clf.coef-)
print('beta0:', lr_clf.intercept_)

```

4. visualization
```python
plt.figure()
plt.scatter(x_features[:,0]),x_features[:,1], c = y_label, s = 50, cmap = 'viridis'
plt.title('Simulated Dataset')
plt.show()
```

5. showing classification boundary line
```python
plt.figure()
plt.scatter(x_features[:, 0], x_features[:, 1], c = y_label, cmap = 'viridis')
plt.title('Simulated Dataset')

nx, ny = 200, 200
x_min, x_max = plt,xlim()
y_min, y_max = plt.ylim()
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))

z_proba = lr_clf.predict_proba(np.c_[x_grid.ravel(), y_grid.ravel()])
z_proba = z_proba[:, 1].reshape(x_grid,shape)
plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths = 2., colors = 'blue')

plt.show
```

6. visualization test data prediction

```python
plt.figure()
## new point 1
x_features_new1 = np.array([[0, -1]])
plt.scatter(x_features_new1[:,0], x_features_new1[:,1], s = 50, cmap = 'viridis')
plt.annotates(s = 'New Point 1', xy = (0, -1)), xytext = (-2, 0), color = 'blue', arrowprops = dict(arrowstyle = '-|>', connectionstyle = 'arc3', color = 'red'))

## new point 2
x_fearures_new2 = np.array([[1, 2]])
plt.scatter(x_fearures_new2[:,0],x_fearures_new2[:,1], s=50, cmap='viridis')
plt.annotate(s='New point 2',xy=(1,2),xytext=(-1.5,2.5),color='red',arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='red'))

## testing data
plt.scatter(x_fearures[:,0],x_fearures[:,1], c=y_label, s=50, cmap='viridis')
plt.title('Dataset')

# visualizing classification boundary
plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')

plt.show()
```

7. model prediction

```python
## predicting training set and testing set
y_label_new1_predict = lr_clf.predict(x_fearures_new1)
y_label_new2_predict = lr_clf.predict(x_fearures_new2)

print('The New point 1 predict class:\n',y_label_new1_predict)
print('The New point 2 predict class:\n',y_label_new2_predict)

## predicting probability using predict_proba function
y_label_new1_predict_proba = lr_clf.predict_proba(x_fearures_new1)
y_label_new2_predict_proba = lr_clf.predict_proba(x_fearures_new2)

print('The New point 1 predict Probability of each class:\n',y_label_new1_predict_proba)
print('The New point 2 predict Probability of each class:\n',y_label_new2_predict_proba)
```
The New point 1 predict class:
 [0]
The New point 2 predict class:
 [1]
The New point 1 predict Probability of each class:
 [[0.69567724 0.30432276]]
The New point 2 predict Probability of each class:
 [[0.11983936 0.88016064]]


[click me for markdown learning](https://www.youtube.com/watch?v=eJojC3lSkwg&t=105s)

![pics](https://picsome.photos)

```Python
y = x = 1
print(x+y)
```


|head| head|head|
|---|---|---|
|a|b|c|
|v|c|s|
|f|c|d|
|r|d|3|


~~not easy~~
