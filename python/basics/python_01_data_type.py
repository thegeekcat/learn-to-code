# -*- coding: utf-8 -*-
"""Python-01 Data Type.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VP_fDfjxtFjzRBZYAotcnjZGZxxOzJGC

# Text Type 1
"""

print ('Hello World!')

# Rules of Variable
a = 1
b = 2

# 2a = 3 (X) -> Variables need to start with letters
student_count = 101
print(a * b, student_count)

friend = 1
Friend = 10

print ("friend: ", friend)
print ("Friend: ", Friend)

"""# Numeric Types"""

year = 2023
month = 4

print(year, month)

year = 'Test'

print(year)

print(type(year))

"""## Number Systems"""

# Octal: 0o + int

print(0o10)

# Binary: 0b + int
print (0b10)

# Hexadecimal: 0x + int
print(0x10)

# Ways to convert decimal numbers (Result is a str type)


print (oct(38)) # Convert decimal 38 to octal
print (hex(38)) # Convert decimal 38 to Hexadecimal
print (bin(38)) # Convert decimal 38 to Binary

"""## Int """

print (type(1)) # int
print (type(2**31)) # 2^31 is int (no more long type from Python v3.0)

"""## Float """

print(type(3.14))
print(type(3.14e-2))
print(type(3 - 4j)) # Complex number

x = 3 - 4j
print(x.imag) # imag is imaginary part
print(x.real) # real is real part
print(x.conjugate()) #conjugate is conjugate complex number

# Practice: Find the area of the circle and the area of the triangle

r = 2

circle_area = (r ** 2) * 3.14
print(circle_area)

x = 3
y = 4
triangle_area = x * y / 2
print(triangle_area)

"""## Text Type 2"""

print('Hello World!')
print("Hello World!")
print("Welcome to 'Python World'")
print('''
      Do What You Love
''')

print('\t Tab \n Next Line')

print('py' 'thon')
print('py' + 'thon') 
print('py', 'thon')

print('py' * 3)

a = 'Python'

print(a[0])
print(a[0+1])
print(a[0+2])

a[0] = 'A' # Index is 'real-only' so it's unmodifiable

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

"""# Data Type Conversion"""

print(str(3.14)) # Convert to String
print(int('49')) # Convert to Int
print(float(23)) # Convert to Float