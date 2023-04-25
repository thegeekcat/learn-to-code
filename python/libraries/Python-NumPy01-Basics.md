# 1. Basics

**** Note: I refer to many sources from the internet, my classes, and books; including `w3school.com`.

## 1.1. Import NumPy


```python
import numpy as np
```

## 1.2. Check NumPy Version


```python
print(np.__version__)
```

    1.22.4
    



---



# 2. Create Arrays

## 2.1. Create a NumPy `ndarray` 

- `ndarray`
 - Def: The 'Array' in Numpy is called `ndarray`
 - Usage: Name is `ndarray` but it's used as `array()` function


```python
# Create a NumPy `ndarray` using `array()` function

arr = np.array([1, 2, 3, 4, 5])  # list []

print(arr)
print(type(arr))   # Check data type
```

    [1 2 3 4 5]
    <class 'numpy.ndarray'>
    


```python
# Create a NumPy `ndarray` using a 'tuple'

arr = np.array((1, 2, 3, 4, 5))  # tuple ()

print(arr)
print(type(arr))
```

    [1 2 3 4 5]
    <class 'numpy.ndarray'>
    


```python
# Compare a 'list' and a 'tuple'
print('Type of [1, 2, 3]: ', type([1, 2, 3]))
print('Type of (1, 2, 3): ', type((1, 2, 3)))
```

    Type of [1, 2, 3]:  <class 'list'>
    Type of (1, 2, 3):  <class 'tuple'>
    

## 2.2. Dimensions in Arrays

### 0-Dimentional Arrays


```python
arr = np.array(1)

print('0-D Array: ', arr)
```

    0-D Array:  1
    

### 1-Dimentional Arrays


```python
arr = np.array([1, 2, 3, 4, 5])  # Use the list type to make an array

print('1-D Array:', arr)
```

    1-D Array: [1 2 3 4 5]
    

### 2-Dimentional Arrays


```python
arr = np.array([[1, 2], [3, 4]]) # Use the list type

print('2-D Array: \n', arr)
```

    2-D Array: 
     [[1 2]
     [3 4]]
    

### 3-Dimentional Arrays


```python
arr = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
               [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])

print('3-D Array: \n', arr)
```

    3-D Array: 
     [[[ 1  2  3  4  5]
      [ 6  7  8  9 10]]
    
     [[11 12 13 14 15]
      [16 17 18 19 20]]]
    

### Higher Dimensional Arrays


```python
# `ndmin`: Define the number of dimentions

arr = np.array([1, 2, 3], ndmin=5)

print(arr)
print('The number of dimensions: ', arr.ndim)
```

    [[[[[1 2 3]]]]]
    The number of dimensions:  5
    


```python
arr = np.array
```



---



# 3. Array Indexing



- NumPy Indexing
 - Starting with '0'

## 3.1. Access Array Elements

### Access Array Elements in 1-D Arrays


```python
arr = np.array([1, 2, 3, 4])  # list type

print(arr[0])  # Access Index[0]='1'
print(arr[3])  # Access Index[3]='4'
```

    1
    4
    

### Access Array Elements in 2-D Arrays


```python
arr = np.array([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10]])

print('4th element on 2nd row: ', arr[0, 1])  # 4th on 2nd row='9'
```

    4th element on 2nd row:  2
    

### Access Array Elements in 3-D Arrays


```python
arr = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
               [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
               
print(arr.ndim)
print(arr)

print('Array[1, 0, 2]: ', arr[1, 0, 2])
```

    3
    [[[ 1  2  3  4  5]
      [ 6  7  8  9 10]]
    
     [[11 12 13 14 15]
      [16 17 18 19 20]]]
    Array[1, 0, 2]:  13
    

## 3.2. Negative Indexing

- Negative Indexing: Access an array from the end
                    -> Using 'Minus'  e.g. -1, -2


```python
arr = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]) # list

print(arr)

print('\nThe num of dimentions: ', arr.ndim)

print('\nThe second last element from 1st dimention: ', arr[0, -2])
```

    [[ 1  2  3  4  5  6  7  8  9 10]
     [11 12 13 14 15 16 17 18 19 20]]
    
    The num of dimentions:  2
    
    The second last element from 1st dimention:  9
    



---



# 4. Array Slicing

- [start index : end index-1 :  step]
  * Note: the end index is excluded 

## 4.1. Slice Arrays


```python
# Example 1

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # list

print(arr[3:5])  # from Index 3('4') to a number before Index 5('6') => excluding the end index
```

    [4 5]
    


```python
# Example 2

arr = np.array([1, 2, 3, 4, 5]) #list

print(arr[0:3]) # between index 0 and index 2 (index 3 is excluded) 
```

    [1 2 3]
    


```python
# Example 3

arr = np.array([1, 2, 3, 4, 5]) #list

print(arr[:4]) # from the beginning to index3 -> 1, 2, 3, 4
```

    [1 2 3 4]
    


```python
# Example 4

arr = np.array([1, 2, 3, 4, 5]) #list

print(arr[3:]) # from index 3 to the end -> 4, 5
```

    [4 5]
    

## 4.2. Negative Slicing 

- **Negative slicing**: slice from or to index from the end


```python
# Example 1

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) #list

print(arr[3:-1]) # from index 3 to index 9 -> 4, 5, 6, 7, 8, 9
```

    [4 5 6 7 8 9]
    


```python
# Example 2

arr = np.array([1, 2, 3, 4, 5, 6, 7])  # list

print(arr[-4:-2]) # from index 3 to index 6 -> 4, 5
```

    [4 5]
    

## 4.3. Step


```python
# Excercise 1
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) #list

print(arr[2:7:2])  # from the beginning to the end by 2 -> 1, 3, 5, 7, 9
```

    [3 5 7]
    


```python
# Excercise 2
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) #list

print(arr[::3])  # from the beginning to the end by 2
```

    [ 1  4  7 10]
    


```python
# ## 4.4. Slicing 2-Dimentioanl Arrays
```


```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]]) # list

print(arr,'\n')

print(arr[1,    # index 1('2')
          0:2]) # from the beginning to index 1 -> 4, 5
```

    [[1 2 3]
     [4 5 6]] 
    
    [4 5]
    



---



# 5. Data Type

- Data Type in **Python**
 - `strings`  e.g. a, b, c
 - `integer`  e.g. 1, 2, 3
 - `float`    e.g. 0.5, 2.4
 - `boolean`  e.g. True, False
 - `complex`  e.g. 0.3+1.1j

- Data Type in **NumPy**
 - `S`: string
 - `U`: unicode string
 - `i`: integer
 - `u`: unsigned integer
 - `f`: float
 - `b`: boolean
 - `c`: complex float

 - `m`: timedelta
 - `M`: datetime

 - `O`: object


## 5.1. Check Data Types


```python
arr.dtype
```




    dtype('int64')




```python
print(arr.dtype)
```

    int64
    

## 5.2. Create Arrays with a defined Data Type


```python
# Example 1

arr = np.array([1, 2, 3],  # list
               dtype='S')  # S: string

print(arr)
print('Data Type: ', arr.dtype)
```

    [b'1' b'2' b'3']
    Data Type:  |S1
    


```python
# Example 2

arr = np.array([1, 2, 3],  # list
               dtype='i4')

print(arr)
print('Data Type: ', arr.dtype)
```

    [1 2 3]
    Data Type:  int32
    



---



# 6. Copy and View

- Copy and View
 - Copy: Make a new array from the original array
        -> copied data won't affect original array
 - View: Make a view of the original array
        -> the orriginal array will be affected when data in view is changed

## 6.1. Copy

- **Copy**: arr.copy()


```python
# Make an array + Copy the array
arr = np.array([1, 2, 3, 4, 5]) # list
arr_copy = arr.copy()  # Make a copy from the original array

print('arr: ', arr)
print('arr_copy: ', arr_copy)

# Make a change in the original array
arr[0] = 50
print('\narr: ', arr) # result: arr[0] is changed
print('arr_copy: ', arr_copy) # copied array is an independent array, so it's not affected by the change of original array
```

    arr:  [1 2 3 4 5]
    arr_copy:  [1 2 3 4 5]
    
    arr:  [50  2  3  4  5]
    arr_copy:  [1 2 3 4 5]
    

## 6.2. View

- **View**: arr.view()


```python
# Make an array + Make a view of the array

arr = np.array([1, 2, 3, 4, 5]) # list
arr_view = arr.view()  # Make a view of the original array

print('arr: ', arr)
print('arr_view: ', arr_view)

# Make a change in the original array
arr[0] = 50
print('\narr: ', arr) # result: arr[0] is changed
print('arr_view: ', arr_view)  # result: the change in the original array is affected to the View
```

    arr:  [1 2 3 4 5]
    arr_view:  [1 2 3 4 5]
    
    arr:  [50  2  3  4  5]
    arr_view:  [50  2  3  4  5]
    

## 6.3. Check `copy()` or `view()`

- Check
 - **copy.base**: return `None`
 - **view.base**: return the original array


```python
arr = np.array([1, 2, 3, 4, 5]) # list

arr_copy = arr.copy()
arr_view = arr.view()

print(arr_copy.base)  #
print(arr_view.base)
```

    None
    [1 2 3 4 5]
    



---



# 7. Array Reshape

## 7.1. Shape Arrays

- **Shape**: the number of elements in each dimension

 - arr.**shape**
 - result: (the number of columns, the number of rows)


```python
# Example 1

arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape   # result: (the number of columns, the number of rows) -> (2, 3)
```




    (2, 3)




```python
# Example 2

arr = np.array([1, 2, 3, 4], ndmin=5) 

print(arr)
print('A shape of array: ', arr.shape)
```

    [[[[[1 2 3 4]]]]]
    A shape of array:  (1, 1, 1, 1, 4)
    

## 7.2. Reshape Arrays

- **Reshape**
 - Change the shape of an array
 - Being able to add or remove dimensions
 - Being able to number of elements in each dimension

- Arrays 
 - 2-D Array: arr.**reshape(rows, columns)**
 - 3-D Array: arr.**reshape(arrays, rows, columns)**

- `arr = np.arrange(30).reshape(2, 3, 5)`

### Reshape from 1-D to 2-D


```python
# Make a new array with 12 elements
arr = np.arange(12)  
print('original array: ', arr)

# Reshape `arr`
new_arr = arr.reshape(3,4) # reshape -> 3 rows * 4 columns
print('\n reshaped array: \n', new_arr)
```

    original array:  [ 0  1  2  3  4  5  6  7  8  9 10 11]
    
     reshaped array: 
     [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    

### Reshape from 1-D to 3-D


```python
# Make a new array with 12 elements
arr = np.arange(30)  
print('original array: ', arr)

# Reshape `arr`
new_arr = arr.reshape(2, 3, 5) # reshape -> 2 arrays + 3 rows * 4 columns
print('\n reshaped array: \n', new_arr)
```

    original array:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29]
    
     reshaped array: 
     [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]
      [10 11 12 13 14]]
    
     [[15 16 17 18 19]
      [20 21 22 23 24]
      [25 26 27 28 29]]]
    


```python
# Combine two lines above:

arr = np.arange(30).reshape(2, 3, 5)  # TADA!
arr
```




    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]],
    
           [[15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29]]])



### Check a type of returned arrays

- `arr.reshape(2,4).base`


```python
arr = np.arange(15).reshape(3, 5)

print('arr: \n', arr)

print('\ntype : ', arr.base) # result: it returned the original array, so it means it's a 'VIEW'
```

    arr: 
     [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]]
    
    type :  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
    

### Unknown Dimension

- use `-1`


```python
# Use `-1` for an unknown columns

new_arr = np.arange(36).reshape(3, 4, -1) # reshape -> 3 arrays + (4 rows * ? columns)
             
print(new_arr)
```

    [[[ 0  1  2]
      [ 3  4  5]
      [ 6  7  8]
      [ 9 10 11]]
    
     [[12 13 14]
      [15 16 17]
      [18 19 20]
      [21 22 23]]
    
     [[24 25 26]
      [27 28 29]
      [30 31 32]
      [33 34 35]]]
    


```python
# Use `-1` for an unknown rows

new_arr = np.arange(36).reshape(3, -1, 4) # reshape -> 3 arrays + (? rows * 4 columns)

print(new_arr)
```

    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
    
     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]
    
     [[24 25 26 27]
      [28 29 30 31]
      [32 33 34 35]]]
    


```python
# Use `-1` for an unknown arrays

new_arr = np.arange(36).reshape(-1, 3, 4) # reshape -> ? arrays + (3 rows * 4 columns)

print(new_arr)
```

    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
    
     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]
    
     [[24 25 26 27]
      [28 29 30 31]
      [32 33 34 35]]]
    



---



# 8. Array Iterating

## 8.1 Interate Arrays

### Iterating 1-D Arrays


```python
arr = np.arange(3)

for x in arr:
  print(x)
```

    0
    1
    2
    

### Iterating 2-D Arrays


```python
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]) # list

for x in arr:
  print(x)
```

    [1 2 3 4 5]
    [ 6  7  8  9 10]
    

### Iterating 3-D Arrays


```python
arr = np.array([[[1, 2, 3],[4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  print(x)
```

    [[1 2 3]
     [4 5 6]]
    [[ 7  8  9]
     [10 11 12]]
    


```python
arr = np.array([[[1, 2, 3],[4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]])

for i in arr:
  for y in x:
    for z in y:
      print(z)
```

    7
    8
    9
    10
    11
    12
    7
    8
    9
    10
    11
    12
    


```python
arr = np.arange(3)

for x in np.nditer(arr,
                   flags=['buffered'],
                   op_dtypes=['S']):
  print(x)
```

    b'0'
    b'1'
    b'2'
    



---



# 9. Join Arrays

- Join Arrays
 - Put two or more arrays in an array
 - Join arrays by axes

 - `concatenate(array, array)`

## 9.1. Join Arrays

### Join 1-D Arrays


```python
# Create two 1-D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Join the arrays
arr = np.concatenate((arr1, arr2))

# Check
arr
```




    array([1, 2, 3, 4, 5, 6])



### Join 2-D Arrays (axis=1)


```python
# Create two 2-D arrays
arr1 = np.array([[1,2,3], [4,5,6]])
arr2 = np.array([[7, 8, 9], [10,11,12]])

# Join the arrays
arr = np.concatenate((arr1, arr2))

# Check
arr
```




    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]])



### Join 3-D Arrays (axis=2)


```python
arr1 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10,11,12]]])
arr2 = np.array([[[31, 32, 33], [34, 35, 36]], [[37, 38, 39], [40,41,42]]])

arr = np.concatenate((arr1, arr2))

arr
```




    array([[[ 1,  2,  3],
            [ 4,  5,  6]],
    
           [[ 7,  8,  9],
            [10, 11, 12]],
    
           [[31, 32, 33],
            [34, 35, 36]],
    
           [[37, 38, 39],
            [40, 41, 42]]])



## 9.2. Stack Arrays along Rows

- Stack arrays along rows
 - `np.hstack(array, array)`


```python
# Make arrays
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])

# Stack arrays
arr = np.hstack((arr1, arr2))

# Check
arr
```




    array([1, 2, 3, 4, 5, 6])



## 9.3. Stack Arrays along Columns

- Stack arrays along columns
 - `np.vstack(array, array)`


```python
# Make arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

# Stack arrays
arr = np.vstack((arr1, arr2, arr3))

# Check
arr
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])



## 9.4. Stack Arrays along Height (depth)

- Stack arrays along height
 - `np.dstack(array, array)`


```python
# Make arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

# Stack arrays
arr = np.dstack((arr1, arr2, arr3))

# Check
arr
```




    array([[[1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]]])





---



# 10. Split Arrays

- Split Arrays
 - One array -> multiple arrays
 - `np.array_split(array, the number of arrays)

## 10.1. Split 1-D Arrays


```python
# Make an array
arr = np.arange(24)

# Split the array to 6 arrays
new_arr = np.array_split(arr, 6)

# Check
new_arr
```




    [array([0, 1, 2, 3]),
     array([4, 5, 6, 7]),
     array([ 8,  9, 10, 11]),
     array([12, 13, 14, 15]),
     array([16, 17, 18, 19]),
     array([20, 21, 22, 23])]




```python
# Make an array
arr = np.array([1, 2, 3, 4, 5, 6])

# Split the array to 3 arrays
new_arr = np.array_split(arr, 3)

# Check
print('The 1st array: ', new_arr[0])
print('The 2nd array: ', new_arr[1])
print('The 3rd array: ', new_arr[2])
```

    The 1st array:  [1 2]
    The 2nd array:  [3 4]
    The 3rd array:  [5 6]
    

## 10.2. Split 2-D Arrays 


```python
# Make an array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

# Split the array to 3 arrays
new_arr = np.array_split(arr, 3) # 6 groups in a single array -> split into 3 arrays

# Check
new_arr
```




    [array([[1, 2, 3],
            [4, 5, 6]]),
     array([[ 7,  8,  9],
            [10, 11, 12]]),
     array([[13, 14, 15],
            [16, 17, 18]])]




```python
# Make an array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

# Split the array to 3 arrays
new_arr = np.array_split(arr, 3, axis=1) # 6 groups in a single array -> split into 3 arrays (axis=1)

# Check
new_arr
```




    [array([[ 1],
            [ 4],
            [ 7],
            [10],
            [13],
            [16]]),
     array([[ 2],
            [ 5],
            [ 8],
            [11],
            [14],
            [17]]),
     array([[ 3],
            [ 6],
            [ 9],
            [12],
            [15],
            [18]])]





---



# 11. Array Search

- Search Arrays
  - `np.where(conditions)`

## 11.1. Search Arrays


```python
# Example 1

# Make an array
arr = np.array([1, 1, 2, 3, 4, 1, 5, 6, 3, 6])  # 10 elements

# Search
x = np.where(arr == 1)

# Check
print(x) # Result: index 0, 1, 5
```

    (array([0, 1, 5]),)
    


```python
# Example 2

# Make an array
arr = np.array([1, 1, 2, 3, 4, 1, 5, 6, 3, 6])  # 10 elements

# Search
x = np.where(arr%3 == 1)

# Check
print(x) # Result: index 0, 1, 4, 5
```

    (array([0, 1, 4, 5]),)
    

## 11.2. Search Sorted Arrays

- Search Sorted Arrays
 - Search an index number where the value is located
 - `np.searchsorted(array, value)`

### Search Sorted Arrays


```python
# Make an array
arr = np.array([1, 2, 3, 4, 5])

# Search
x = np.searchsorted(arr, 3)  # Search an index number where the value is located

# Check
print(x) # Result: index 2
```

    2
    

### Search Multiple Values from Sorted Arrays

- `np.searchsorted(array, [array])`


```python
# Make an array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Search
x = np.searchsorted(arr, [4, 5])

# Check
print(x) # Result: Index 3, 4
```

    [3 4]
    



---


# 12. Sort Arrays

- Sort Arrays
  - `np.sort(array)`

## 12.1. Sort 1-D Arrays


```python
# Sort a `Integer` type of arrays

# Make an array
arr = np.array([2, 4, 1, 6, 3, 0])

# Sort
np.sort(arr)
```




    array([0, 1, 2, 3, 4, 6])




```python
# Sort a `String` type of arrays

# Make an array
arr = np.array(['orange', 'mango', 'Korean melon', 'Tomato'])

# Sort
np.sort(arr)
```




    array(['Korean melon', 'Tomato', 'mango', 'orange'], dtype='<U12')



## 12.2. Sort 2-D Arrays


```python
# Make an array
arr = np.array([[6, 4, 1], [8, 3, 5]])

# Sort
np.sort(arr)
```




    array([[1, 4, 6],
           [3, 5, 8]])





---



# 13. Typs of Arrays: zeros, ones, empty

- Other types of Arrays
 - `np.zero((x, y))`: An array filled with '0'
 - `np.ones((x, y))`: An array filled with '1'
 - `np.empty((x, y))`: An empty array

## 13.1. zeros Arrays


```python
np.zeros((3, 3)) # an array filled with '0'
```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])



## 13.2. ones Arrays


```python
np.ones((2, 2))  # an array filled with '1'
```




    array([[1., 1.],
           [1., 1.]])



## 13.3. empty Arrays


```python
np.empty((4, 4)) # an empty array
```




    array([[1.77997702e-316, 0.00000000e+000, 2.41907520e-312,
            2.37663529e-312],
           [2.22809558e-312, 2.46151512e-312, 6.79038654e-313,
            2.35541533e-312],
           [2.46151512e-312, 6.79038654e-313, 2.35541533e-312,
            6.79038654e-313],
           [2.14321575e-312, 2.22809558e-312, 2.14321575e-312,
            6.93447005e-310]])





---



# 14. Arange

- Arange: Create a list
 - `np.arange(number)`: Make a list with the number of elements
 - `np.arange(start, end-1)`: Make a list from 'start' to 'end-1'
 - `np.arange(start, end-1, step)`: Make a list from 'start' to 'end-1' by 'step'


```python
np.arange(10)  # Make a list with 10 elements -> from 0 to 9
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.arange(1,10)  # Make a list from 1 to 9
```




    array([1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.arange(1, 10, 2) # Make a list from 1 to 9 by 2
```




    array([1, 3, 5, 7, 9])





---



# 15. Calculation of Arrays


```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
print(arr1 + arr2)
print(arr1 - arr2)
print(arr1 * arr2)
```

    [[ 6  8]
     [10 12]]
    [[-4 -4]
     [-4 -4]]
    [[ 5 12]
     [21 32]]
    


```python
print(np.add(arr1, arr2))
print(np.multiply(arr1, arr2))
```

    [[ 6  8]
     [10 12]]
    [[ 5 12]
     [21 32]]
    
