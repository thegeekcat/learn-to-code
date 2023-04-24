- Try.. Except
 - `try`: Test a block of codes for errors
 - `except`: Manage the errors
 - `else`: Excute codes when there's no error
 - `finally`: Excute codes regardless of the result of 'try' and 'except'


```python
def divide(a, b):
  return a / b
```


```python
try:
  c = divide(5, 0)
except:
  print('Exception is occured!')
```

    Exception is occured!
    


```python
try: 
  c = divide('a', 0)
except ZeroDivisionError:
  print('The second value shouldn\'t be \'0\'')
except TypeError:
  print('A type of all values must be Number')
except:
  print('No idea')
finally:
  print('GG')
```

    A type of all values must be Number
    GG
    
