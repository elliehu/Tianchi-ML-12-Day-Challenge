 **NumPy**

NumPy array elements should all be the same type, or rather, it will automatically cast to the same.

```python

import numpy as np 
# nested lists result in multi-dimensional arrays
np.array([range(i, i + 3) for i in [2, 4, 6]])

```

1. **Data Types**
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



2. **Manipulating Array**

- Array slicing
``` python

import numpy as np
np.random.seed(0)

x1 = np.random.randint(10, size = 6)
x2 = np.random.randint(10, size = (3, 4))
x3 = np.random.randint(10, size = (3, 4, 5))

print("x3 ndim:", x3.ndim) # ndim (number of dimensions)
print("x3 shape:", x3.shape) # shape (size of each dimension)
print("x3 size:", x3.size) # total size of array
print("x3 dtype:", x3.dtype)


x1[0] = 3.14 # this will be truncated since numpy array elements need to be of the same type



x[start: stop: step] # default values: start = 0, stop = size of dimension, step =1

x = np.arrange(10) # array([0,1,2,3,4,5,6,7,8,9])

x[:5] # first 5 elements

x[5:] #elements after index 5

x[4 :7] # array([4, 5, 6])

x[::2] #every other element, array([0, 2, 4, 6, 8])

x[1::2] # every other element staring at index 1, array([1, 3, 5, 7, 9])


# When step value is negative, start and stop are swapped:
x[::-1] # all elements reversed

x[5::-2] # array([5, 3, 1])

```

