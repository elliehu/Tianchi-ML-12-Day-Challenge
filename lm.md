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





[click me for md learning](https://www.youtube.com/watch?v=eJojC3lSkwg&t=105s)

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
