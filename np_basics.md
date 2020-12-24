> - **numpy packages**

NumPy array elements should all be the same type, or rather, it will automatically cast to the same.

```python

import numpy as np 
# nested lists result in multi-dimensional arrays
np.array([range(i, i + 3) for i in [2, 4, 6]])

```

```python
# creating integer array filled the zeros
np.zeros(10, dtype = int)

# creating float array filled the ones
np.zeros(1, dtype = float)

# creating a 3X5 array filled with 3,14
np.full((3, 5), 3.14)


# creating sequence
np.arange(0, 20, 2) # array(0,2,4,6,8,10,12,14,16,18)

# creating evenly spaced squence between 0 and 1
np.linspace(0, 1, 5)

# creating 3X3  uniform array of random values between 0 and 1
np.random.random(3, 3)

# creating 3x3 array of normally distributed random values with mean = 0 and std =1
np.random.normal(0, 1, (3, 3))

# creating  3x3 array of random integer in the interval [0, 10)
np.random.randint(0, 10,(3, 3))


# creating 3x3 identity matrix, one-hot encoding?
np.eye(3)



```
