# Text Type 1


```python
print ('Hello World!')
```

    Hello World!
    


```python
# Rules of Variable
a = 1
b = 2

# 2a = 3 (X) -> Variables need to start with letters
student_count = 101
print(a * b, student_count)
```

    2 101
    


```python
friend = 1
Friend = 10

print ("friend: ", friend)
print ("Friend: ", Friend)

```

    friend:  1
    Friend:  10
    

# Numeric Types


```python
year = 2023
month = 4

print(year, month)

year = 'Test'

print(year)

print(type(year))
```

    2023 4
    Test
    <class 'str'>
    

## Number Systems


```python
# Octal: 0o + int

print(0o10)

# Binary: 0b + int
print (0b10)

# Hexadecimal: 0x + int
print(0x10)
```

    8
    2
    16
    


```python
# Ways to convert decimal numbers (Result is a str type)


print (oct(38)) # Convert decimal 38 to octal
print (hex(38)) # Convert decimal 38 to Hexadecimal
print (bin(38)) # Convert decimal 38 to Binary
```

    0o46
    0x26
    0b100110
    

## Int 


```python
print (type(1)) # int
print (type(2**31)) # 2^31 is int (no more long type from Python v3.0)
```

    <class 'int'>
    <class 'int'>
    

## Float 


```python
print(type(3.14))
print(type(3.14e-2))
print(type(3 - 4j)) # Complex number

x = 3 - 4j
print(x.imag) # imag is imaginary part
print(x.real) # real is real part
print(x.conjugate()) #conjugate is conjugate complex number
```

    <class 'float'>
    <class 'float'>
    <class 'complex'>
    -4.0
    3.0
    (3+4j)
    


```python
# Practice: Find the area of the circle and the area of the triangle

r = 2

circle_area = (r ** 2) * 3.14
print(circle_area)

x = 3
y = 4
triangle_area = x * y / 2
print(triangle_area)
```

    12.56
    6.0
    

## Text Type 2


```python
print('Hello World!')
print("Hello World!")
print("Welcome to 'Python World'")
print('''
      Do What You Love
''')
```

    Hello World!
    Hello World!
    Welcome to 'Python World'
    
          Do What You Love
    
    


```python
print('\t Tab \n Next Line')
```

    	 Tab 
     Next Line
    


```python
print('py' 'thon')
print('py' + 'thon') 
print('py', 'thon')
```

    python
    python
    py thon
    


```python
print('py' * 3)
```

    pypypy
    


```python
a = 'Python'

print(a[0])
print(a[0+1])
print(a[0+2])
```

    P
    y
    t
    


```python
a[0] = 'A' # Index is 'real-only' so it's unmodifiable
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-71-edede21216ea> in <cell line: 1>()
    ----> 1 a[0] = 'A' # Index는 real-only속성이라 수정은 불가능함
    

    TypeError: 'str' object does not support item assignment



```python
# Set a range using index
print(a[0:1])
print(a[0:2])

print(a[:3]) # ommittable
print(a[3:]) 

# P y t h o n
# 0 1 2 3 4 5

print(a[-3:]) # Only the three digits at the end  e.g. for Product no
print(a[:-3]) # Only the three digits at the beginning

print(a[::2]) # increasing by 2
```

    P
    Py
    Pyt
    hon
    hon
    Pyt
    Pto
    

# Data Type Conversion


```python
print(str(3.14)) # Convert to String
print(int('49')) # Convert to Int
print(float(23)) # Convert to Float
```

    3.14
    49
    23.0
    
