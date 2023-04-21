# Lists

## List Methods
- **append()**:	Adds an element at the end of the list
- **clear()**:	Removes all the elements from the list
- **copy()**:	Returns a copy of the list
- **count()**:	Returns the number of elements with the -specified value
-** extend()**:	Add the elements of a list (or any iterable), to the end of the current list
- **index()**:	Returns the index of the first element with the specified value
- **insert()**:	Adds an element at the specified position
- **pop()**:	Removes the element at the specified position
- **remove()**:	Removes the item with the specified value
- **reverse()**:	Reverses the order of the list
- **sort()**:	Sorts the list

*(Note: These explains of methods above refers to w3schools.com; but not the examples below.)*



```python
# Basics
colors = ['red', 'green', 'gold']
print(colors)
print(type(colors))
```

    ['red', 'green', 'gold']
    <class 'list'>
    

## Add List Items

### Append Items


```python
# 'append()' method: to add an item to the end of the list
colors.append('blue') 
print(colors)
```

    ['red', 'green', 'gold', 'blue']
    

### Insert Items


```python
# 'insert()' method: to insert a list item at a specified index
colors.insert(1, 'black') # Insert 'black' into index 1
print(colors)
```

    ['red', 'black', 'green', 'gold', 'blue']
    

### Extend List


```python
# 'expend()' method1: to append items from another list to the current list
colors = ['red', 'blue', 'black']
color2 = ['orange', 'gold', 'yellow']

colors.extend(color2)  
print("First expend: ", colors) 

colors.extend(['purple', 'green'])
print("Second expend: ", colors)
```

    First expend:  ['red', 'blue', 'black', 'orange', 'gold', 'yellow']
    Second expend:  ['red', 'blue', 'black', 'orange', 'gold', 'yellow', 'purple', 'green']
    


```python
colors += ['red'] # (with []): considred as a single word
print(colors)

colors += 'yellow' # (without []): considered each words as one item
print(colors)
```

    ['red', 'blue', 'black', 'orange', 'gold', 'yellow', 'purple', 'green', 'red']
    ['red', 'blue', 'black', 'orange', 'gold', 'yellow', 'purple', 'green', 'red', 'y', 'e', 'l', 'l', 'o', 'w']
    

## index() method


```python
print(colors) # Two 'red' words in the list

print(colors.index('red')) # Find an index where the word is located from the beginning 

print(colors.index('red', 1)) #Find an index where the word is located from index 1
```

    ['red', 'blue', 'black', 'orange', 'gold', 'yellow', 'purple', 'green', 'red', 'y', 'e', 'l', 'l', 'o', 'w']
    0
    8
    

## pop() method


```python
colors.pop() # 
```




    'w'




```python
colors.pop(1)
```




    'blue'




```python
print(colors)
```

    ['red', 'black', 'orange', 'gold', 'yellow', 'purple', 'green', 'red', 'y', 'e', 'l', 'l', 'o']
    

## Remove Lists 


```python
colors.remove('e') # Remove the specific item
print(colors)
```

    ['red', 'black', 'orange', 'gold', 'yellow', 'purple', 'green', 'red', 'y', 'l', 'l', 'o']
    

## Sort Lists

### Sort Lists Alphanumerically


```python
colors.sort()
print(colors)
```

    ['black', 'gold', 'green', 'l', 'l', 'o', 'orange', 'purple', 'red', 'red', 'y', 'yellow']
    

### Sort Lists Descending


```python
colors.reverse()  # Available option:  colors.sort(reverse=True)
print(colors)
```

    ['yellow', 'y', 'red', 'red', 'purple', 'orange', 'o', 'l', 'l', 'green', 'gold', 'black']
    

# Sets

## Set Methods


- **add()**:	Adds an element to the set
- **clear()**:	Removes all the elements from the set
- **copy()**:	Returns a copy of the set
- **difference()**:	Returns a set containing the difference between two or more sets
- **difference_update()**:	Removes the items in this set that are also included in another, specified set
- **discard()**:	Remove the specified item
- **intersection()**:	Returns a set, that is the intersection of two other sets
- **intersection_update()**:	Removes the items in this set that are not present in other, specified set(s)
- **isdisjoint()**: Returns whether two sets have a intersection or not
- **issubset()**: Returns whether another set contains this set or not
- **issuperset()**: Returns whether this set contains another set or not
- **pop()**:	Removes an element from the set
- **remove()**:	Removes the specified element
- **symmetric_difference()**:	Returns a set with the symmetric differences of two sets
- **symmetric_difference_update()**:	inserts the symmetric differences from this set and another
- **union()**:	Return a set containing the union of sets
- **update()**:	Update the set with the union of this set and others

*(Note: These explains of methods above refers to w3schools.com; but not the examples below.)*




```python
# Advantage 1: Using less memory than 'List'
a = {1, 2, 3}
b = {3, 4, 5}

print(a, b)

print(type(a))
```

    {1, 2, 3} {3, 4, 5}
    <class 'set'>
    


```python
# # Advantage 2: 'Union' and 'Intersection' available 

print ('union: ', a.union(b)) 
print ('intersection: ', a.intersection(b))
```

    union:  {1, 2, 3, 4, 5}
    intersection:  {3}
    


```python
print (a-b) # Subtraction
print (a|b) # Union
print(a&b) # Intersection
```

    {1, 2}
    {1, 2, 3, 4, 5}
    {3}
    

# Tuples

## Tuple Methods

- **count()**:	Returns the number of times a specified value occurs in a tuple
- **index()**	Searches the tuple for a specified value and returns the position of where it was found

*(Note: These explains of methods above refers to w3schools.com; but not the examples below.)*


```python
t = (1, 2, 3)
print(type(t))
```

    <class 'tuple'>
    


```python
(a, b) = (1, 2)
print(a, b)
```

    1 2
    

# Excercises 

## Change items 


```python
# Switching variables

a = 10
b = 20
(a, b) = (b, a)
print(a, b)
```

    20 10
    


```python
# Changing types
a = set ((1, 2, 3))
print(type(a))

b = list(a)
print(type(b))

c = tuple(a)
print(type(c))

d = set(c)
print(type(d))
```

    <class 'set'>
    <class 'list'>
    <class 'tuple'>
    <class 'set'>
    

## 'in' operator


```python
# Use 'in' operator to check a value in a Set
a = set ((1, 2, 3))
print(a)
print (1 in a) # Check whether '1' is in Set 'a'

print(5 in a)
```

    {1, 2, 3}
    True
    False
    

## Dictionary 


```python
# Store values in 'key:value' pairs
d = dict(a = 1, b = 2, c = 3)  
print(d)
print(type(d))
```

    {'a': 1, 'b': 2, 'c': 3}
    <class 'dict'>
    


```python
color = {'apple': 'red', 'banana': 'yellow'} # key: value
print(color)
```

    {'apple': 'red', 'banana': 'yellow'}
    


```python
# Errors 
# print(color[0]) # Not available -> dict() has no index
```


```python
# Add a new pair
color['cherry'] = 'red'  
print(color)
```

    {'apple': 'red', 'banana': 'yellow', 'cherry': 'red'}
    


```python
# Change a value in the existing pair
color['apple'] = 'green'  
print(color)
```

    {'apple': 'green', 'banana': 'yellow', 'cherry': 'red'}
    


```python
print(color.keys()  ) # Get 'keys' only
print(color.values()) # Get 'values' only

print(color.items())  # Get 'items'
```

    dict_keys(['apple', 'banana', 'cherry'])
    dict_values(['green', 'yellow', 'red'])
    dict_items([('apple', 'green'), ('banana', 'yellow'), ('cherry', 'red')])
    


```python
# Use 'For Loops' to get data

for k in color.keys():
  print(k)

for v in color.values():
  print(v)

for i in color.items():
  print (i)

for (k, v) in color.items():
  print(k, v)
```

    apple
    banana
    cherry
    green
    yellow
    red
    ('apple', 'green')
    ('banana', 'yellow')
    ('cherry', 'red')
    apple green
    banana yellow
    cherry red
    


```python
# Remove an item from Dictionary
del color['cherry']
print(color)
```

    {'apple': 'green', 'banana': 'yellow'}
    


```python
# Clear itmes in a dictionary
color.clear()
print(color)
```

    {}
    

## Other Excercises


```python
# Mixed use of Data Types
d = { 'age': 40.5, 'job': [1,2,3], 'name': {'Kim':2}, 'cho': 1}

print(d)
print(type(d))
```

    {'age': 40.5, 'job': [1, 2, 3], 'name': {'Kim': 2}, 'cho': 1}
    <class 'dict'>
    


```python
# Boolean

isRight = False
print(type(isRight))
```

    <class 'bool'>
    


```python
print(1 < 2)
print(1 == 2)
print(1 != 2)
```

    True
    False
    True
    


```python
# Logical Operators

print(True and False)
print(True & True)
print(True or False)
print(False | False)
print(not True)
```

    False
    True
    True
    False
    False
    

## 'Call by Reference' and 'Call by Value' 


```python
# Call by Reference
a = [1, 2, 3]
b = a # store 'a' in list 'b' as Call by Reference 
print(b)
```

    [1, 2, 3]
    


```python
a[0] = 38
print(a)
print(b) 
```

    [38, 2, 3]
    [38, 2, 3]
    


```python
# Call by Value
a = [1, 2, 3]
b = a[:] # call by value
a[0] = 38
print(a, b)
```

    [38, 2, 3] [1, 2, 3]
    

### Copy() Module


```python
import copy

a = [1, 2, 3]
b = copy.deepcopy(a) #
a[0] = 28
print(a, b)
```

    [28, 2, 3] [1, 2, 3]
    
