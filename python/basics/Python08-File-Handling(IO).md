# File Handling

- open(filename, mode)
 - `filename` function
    - "r": Read -> returns an error if the file already exists
    - "a": Append -> create a file if it doesn't exist
    - "w": Write -> create a file if it doesn't exist
    - "x": Create -> returns an error if the file already exists
 - `mode` fundtion
    - "t": Text
    - "b": Binary mode  e.g. images





# Create a New File


```python
# Import a module
import sys

# Just printing a sentence
print('Welcome to', 'Python', sep = ' ~ ', end = '!', file = sys.stderr)  # sep(seperation), end(end)
```

    Welcome to ~ Python!


```python
# Create a file

f = open('test.txt', 'w')  # open a txt file in 'w' mode

print('Welcome to Python \n Welcome to ML', file = f)  # print a sentence in a file named 'f'

f.close()  # Close the file 'f'
```


```python
help(sys.__displayhook__)

```

    Help on built-in function displayhook in module sys:
    
    displayhook(object, /)
        Print an object to sys.stdout and also save it in builtins._
    
    

# Read Files


```python
# Read a file

f = open ('test.txt', 'r') # Read Mode

#f.read()  # Read whole texts -> if there's no more text to read, then it shows '' only

#f.readline()   # Read texts by line

f.readline(23)  # Read the number of texts
```




    'Welcome to Python \n'



# Delete Files



```python
# Import a OS module

import os
```


```python
# Delete a File

file_path = './test.txt'

if os.path.exists(file_path):
  os.remove(file_path)
  print("File deleted successfully!")
else:
  print("File doesn't exist.")
```

    File deleted successfully!
    
