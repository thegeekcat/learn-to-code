# 'if' Statement: Conditional Statement


```python
value = 10
if value > 5:
  print('value is bigger than 5')
  
```

    value is bigger than 5
    


```python
money = 10
if money > 100:
  item = 'apple'
else:
  item = 'banana'

print(item)
```

    banana
    


```python
# Excercise: Grading System

score = int(input(('Input your score: ')))  # The value input is a 'str' type, so it needs to be converted to 'int'
        
print("Your score is ", score)

if 90 <= score <= 100:
  grade = 'A'
elif 80 <= score < 90:
  grade = 'B'
elif 70 <= score < 80:
  grade = 'C'
elif 60 <= score < 70:
  grade = 'D'
else:
  grade = 'F'


print('Your score is ' + str(score))  # 'Score' has been converted to 'int' above, so it needs to be 'str' again to print out
print('Your grade is ' + grade)
```

    Input your score: 93
    Your score is  93
    Your score is 93
    Your grade is A
    

# Boolean 


```python
print(bool(True))
print(bool(False))
print(bool(123)) 

print(bool(-1)) 
print(bool(0))
print(bool(''))  # This is a 'str' type but no value -> False
print(bool())    # This is empty -> False
```

    True
    False
    True
    True
    False
    False
    False
    

# 'for' Statement 


```python
l = [ 'Apple', 100, 15.23 ]

for i in l:
  print(i)
```

    Apple
    100
    15.23
    

## Example: Multiplication Tables


```python
# 2 Times Table
loop1 = [2, 3, 4, 5, 6, 7, 8, 9]
loop2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for i in loop2:
  print("2 * " + str(i) + " = " + str(3 * i))
```

    2 * 1 = 3
    2 * 2 = 6
    2 * 3 = 9
    2 * 4 = 12
    2 * 5 = 15
    2 * 6 = 18
    2 * 7 = 21
    2 * 8 = 24
    2 * 9 = 27
    


```python
# 2 Times Table
loop1 = [2, 3, 4, 5, 6, 7, 8, 9]
loop2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for i in loop2:
  print("2 * {0} = {1}".format(i, i*2))
```

    2 * 1 = 2
    2 * 2 = 4
    2 * 3 = 6
    2 * 4 = 8
    2 * 5 = 10
    2 * 6 = 12
    2 * 7 = 14
    2 * 8 = 16
    2 * 9 = 18
    


```python
# 1 - 9 Times Tables

loop1 = [2, 3, 4, 5, 6, 7, 8, 9]
loop2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for i in loop1:
  for j in loop2:
    print("{0} * {1} = {2}".format(i, j, i*j))
```

    2 * 1 = 2
    2 * 2 = 4
    2 * 3 = 6
    2 * 4 = 8
    2 * 5 = 10
    2 * 6 = 12
    2 * 7 = 14
    2 * 8 = 16
    2 * 9 = 18
    3 * 1 = 3
    3 * 2 = 6
    3 * 3 = 9
    3 * 4 = 12
    3 * 5 = 15
    3 * 6 = 18
    3 * 7 = 21
    3 * 8 = 24
    3 * 9 = 27
    4 * 1 = 4
    4 * 2 = 8
    4 * 3 = 12
    4 * 4 = 16
    4 * 5 = 20
    4 * 6 = 24
    4 * 7 = 28
    4 * 8 = 32
    4 * 9 = 36
    5 * 1 = 5
    5 * 2 = 10
    5 * 3 = 15
    5 * 4 = 20
    5 * 5 = 25
    5 * 6 = 30
    5 * 7 = 35
    5 * 8 = 40
    5 * 9 = 45
    6 * 1 = 6
    6 * 2 = 12
    6 * 3 = 18
    6 * 4 = 24
    6 * 5 = 30
    6 * 6 = 36
    6 * 7 = 42
    6 * 8 = 48
    6 * 9 = 54
    7 * 1 = 7
    7 * 2 = 14
    7 * 3 = 21
    7 * 4 = 28
    7 * 5 = 35
    7 * 6 = 42
    7 * 7 = 49
    7 * 8 = 56
    7 * 9 = 63
    8 * 1 = 8
    8 * 2 = 16
    8 * 3 = 24
    8 * 4 = 32
    8 * 5 = 40
    8 * 6 = 48
    8 * 7 = 56
    8 * 8 = 64
    8 * 9 = 72
    9 * 1 = 9
    9 * 2 = 18
    9 * 3 = 27
    9 * 4 = 36
    9 * 5 = 45
    9 * 6 = 54
    9 * 7 = 63
    9 * 8 = 72
    9 * 9 = 81
    

## 'break' Statement


```python
L = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```


```python
for i in L:
  if i > 5:
    break
  print('Item: {0}'.format(i))
```

    Item: 1
    Item: 2
    Item: 3
    Item: 4
    Item: 5
    

## 'continue' Statement


```python
for i in L:
  if i % 2 == 0:   # odd number
    continue
  print('item {0}'.format(i))
```

    item 1
    item 3
    item 5
    item 7
    item 9
    

# 'range()' Function


```python
# range() function returens a range of numbers
# Default: starting from 0, increasing by 1

print(list(range(10)))   # Make a list with 10 numbers starting from '0'
print(list(range(5,10))) # range(start value, end value)
print(list(range(10, 0, -1)))  #range(start value, end value, increasement)
print(list(range(0, 10, 2)))
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    [5, 6, 7, 8, 9]
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    [0, 2, 4, 6, 8]
    


```python
L = ['Apple', 'Orage', 'Banana']

for i in L:
  print(i)
```

    Apple
    Orage
    Banana
    

## A special case of 'for stament' only in Python 


```python
l = list(range(1, 6))
```


```python
# This type of 'for' statment is not existing in other programming languages

[print(i) for i in l]  
# After finishing a cycle of 'for' statement,
# it starts again from the beginning, 
# so its result shows a list of 'None'
```

    1
    2
    3
    4
    5
    




    [None, None, None, None, None]




```python
[i**2 for i in l]
```




    [1, 4, 9, 16, 25]



# 'while' Statement 


```python
value = 5

while value > 0:
  print(value)
  value -= 1 
```

    5
    4
    3
    2
    1
    
