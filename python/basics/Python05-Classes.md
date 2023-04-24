# Basics

## Pass Statement

`Pass` Statement: To create a class without contents


```python
# Excercise 1
class MyClass:
  pass 

print(MyClass)
print(id(MyClass)) # Check memory address
```

    <class '__main__.MyClass'>
    63961136
    

## Override


```python
# Create a class
class Car:
  colors='white'
  brands='Benz'

# Create instances of 'Car'
car1 = Car()
car2 = Car()
car3 = Car()

print('car1.colors: ', car1.colors)
print('car1.brands: ', car1.brands)
print('car2.colors: ', car2.colors)
print('car2.brands: ', car2.brands)
print('car3.colors: ', car3.colors)
print('car3.brands: ', car3.brands)
```

    car1.colors:  white
    car1.brands:  Benz
    car2.colors:  white
    car2.brands:  Benz
    car3.colors:  white
    car3.brands:  Benz
    


```python
# Override
car2.brands='BMW'
print('car2.brands: ', car2.brands)
```

    car2.brands:  BMW
    

# _ _ init _ _() Method

The __init__ method: 
- All classes have the __init__ function
- It's excuted when the cass is initiated




```python
# Class {name_of_class}
#   def __init__(self, args):
#     pass

```


```python
# Create a class

class Cats:
  def __init__(self, name, breed, age):   # `self` parameter: to represent the instance belongs to the class
    self.name = name
    self.breed = breed
    self.age = age
```


```python
# Create an instance
myCat = Cats('Meow', 'Korean Short Hair', '3') # -> call __init__ method when an instance is created
```

# Methods

- Characteristics of Method
 - Being defined inside of Classes
 - Methods are always called by instances  
   : Hiden arguments `self` are passed to methods



```python
# Create a class

class Meow:

  # Define the class itself using '__init__'
  def __init__(self, prefix):
    self.prefix = prefix

  def print_me(self, a, b, c):
    print(self.prefix, a, sep=' _ ')
    print(self.prefix, b, sep=' ~ ')
    print(self.prefix, c, sep=' ! ')
```


```python
# Make an instance
printer = Meow('meeeeooww')

# Call a method
printer.print_me(1, 2, 3)
```

    meeeeooww _ 1
    meeeeooww ~ 2
    meeeeooww ! 3
    


```python
# Excercise 

class Person:
  
  def __init__(self, name, age): 
    self.Name = name
    self.Age = age
  def __del__(self):  
    print('{0} was here'.format(self.Name))


  def PrintName(self):
    print(self.Name)
    print(self.Age)

p = Person('Karen', 20)
p.PrintName()
del p
```

    Karen
    20
    Karen was here
    

# Inheritance

## Basic Inheritance - Excercise 1


```python
# create a Parent class `Mammal`

class Mammal:
  def __init__(self, name, size):
    self.name = name
    self.size = size

  def speak(self):
    print('Yo, I"m ', self.name)

  def call_out(self):
    self.speak()
    self.speak()
```


```python
# Create a child class `Dog`

class Dog(Mammal):
  def speak(self):
    print('Woof! Woof!')
```


```python
# Create a child class `Cat`

class Cat(Mammal):
  def speak(self):
    print('Meeooow!! I"m ', self.name, '! Can you bark??')
```


```python
# Make an instance

my_cat = Cat('Karen', 20)

my_cat.speak()
my_cat.call_out()

#print(my_cat.speak())
#print(my_cat.call_out())
```

    Meeooow!! I"m  Karen ! Can you bark??
    Meeooow!! I"m  Karen ! Can you bark??
    Meeooow!! I"m  Karen ! Can you bark??
    

## `super()` function

`super()` function: to call a method of a parent class


```python
# Create a class with 'super()' function

class BigCat(Mammal):
  def __init__(self, name, size, color):
    super().__init__(name, size)    # call the __init__ from 'Mammal' parent class
    self.color = color

  def speak(self):
    super().speak()
    print('Meooow!', self.name)


# Make an instance
my_bigcat = BigCat('BigMeow', 150, 'Black')

print(my_bigcat.name)
print(my_bigcat.size)
print(my_bigcat.color)

my_bigcat.speak()
```

    BigMeow
    150
    Black
    Yo, I"m  BigMeow
    Meooow! BigMeow
    

## Multiple Inheritance - Excercise 2 



```python
# Multiple Inheritance

# class Class_Name (Upper_class1, Upper_class2):
#    pass
```


```python
# Make a class with multiple inheritance

# -----------------------------------------------------------------------------
# class Pet(Mammal, Cat, BigCat):
#   def __init__(self, name, size, color, characteristic):
#    Mammal.__init__(self, name, size)
#    Cat.__init__(self, name)
#    BigCat.__init__(self, name, size, color)
#    self.characteristic = characteristic

# An error shows: 'Cannot create a consistent method resolution order (MRO) for bases Mammal, Cat, BigCat'

# Reason: 'Cat' and 'BigCat' are inherited from 'Mammal' 

# Solution: Use a diamond inheritance pattern like below:
#     Mammal
#    /     \
#   /       \
# Cat       BigCat
#   \       /
#    \     /
#     Pet

# -----------------------------------------------------------------------------
# Second try

# class Pet(Cat, BigCat):
#  def __init__(self, name, color, size, characteristic):
#    Cat.__init__(self, name)
#    BigCat.__init__(self, color)
#    self.size = size
#    self.characteristic = characteristic
#    
#  def speak(self):
#    super().sleak()
#    print(' and I\'m a cute ', self.name)

# An error shows: __init__() missing 1 required positional argument: 'size'

# -----------------------------------------------------------------------------
# Third try

# class Pet(Cat, BigCat):
#  def __init__(self, name, color, size, characteristic):
#    super().__init__(name)
#    super(BigCat, self).__init__(size)
#    super(Cat, self).__init__(color)
#    self.characteristic = characteristic
#
#  def speak(self):
#    super().speak()
#    print(' and I\'m a cute ', self.name)

# An error shows: __init__() missing 2 required positional arguments: 'size' and 'color'
    
# -----------------------------------------------------------------------------
# 4th try


class Pet(Cat, BigCat):
  def __init__(self, name, size, color, characteristic):
    Cat.__init__(self, name, size=size)  # Issue is here 'size = size'
    BigCat.__init__(self, name=name, size=size, color=color)  # name = name
    self.characteristic = characteristic

  def speak(self):
    super().speak()
    print(' and I\'m a cute ', self.name)



mypet = Pet('Karen_Pet', 'White_Pet', 15, 'cute')

mypet.speak()

```

    Meeooow!! I"m  Karen_Pet ! Can you bark??
     and I'm a cute  Karen_Pet
    


```python

```

# Excercise 1


```python
# 'self' Parameter

class Person:       # Names of Class should be starting with Capital letter  e.g. Person, Print
  Name = 'Default Name'

  def Print(self):   # 'self': to represent the instance belongs to the class
    print('My Name is {0}'.format(self.Name))
```


```python
p1 = Person()
```


```python
p1.Name = 'Karen'
p1.Print()
```

    My Name is Karen
    


```python
p1 = Person()
p2 = Person()
```


```python
p1.Name = 'Bell'
p2.Name = 'Christine'

p1.Print()
p2.Print()
```

    My Name is Bell
    My Name is Christine
    


```python
Person.Title = 'New Title'   
```


```python
print(p1.Title)
print(p2.Title)
```

    New Title
    New Title
    


```python
p1.Age = 30  
```


```python
print(p1.Age)
```

    30
    

# Excercise 2: Inheritance 


```python
# Create a Parent Class
class Person: 
  pass
```


```python
# Create a Child Class
class Student(Person):   # 'Student' class is created as a child class inherited from the parent 'Person'
  pass
```


```python
class Bird:
  pass
```


```python
p = Person() 
s = Student()
```


```python
# Check a relationship between instances

print(isinstance(p, Person))  # Check whether 'p' is inherited from 'Perspn
print(isinstance(s, Person))
print(isinstance(s, object))  # 'object' is the Parent class for all classes in Python
print(isinstance(int, object))
```

    True
    True
    True
    True
    

# Excercise 3: 'Static' Method


```python
class CounterManager:
  InsCount = 0

  def __init__(self):
    CounterManager.InsCount += 1
  def PrintInstanceCount():  
    print('Instance Count:', CounterManager.InsCount)

  StaticCount = staticmethod(PrintInstanceCount)
```


```python
a = CounterManager()
b = CounterManager()
c = CounterManager()
```


```python
CounterManager.PrintInstanceCount() # Be not allowed to use 'Class' directly
```

    Instance Count: 3
    


```python
p = CounterManager()
#p.PrintInstanceCount()
p.StaticCount()
```

    Instance Count: 4
    

# Excercise 4


```python
# Create a Parent class PERSON
class Person:

  def __init__(self, name, phone): # set default values
    self.Name = name
    self.Phone = phone

  def PrintInfo(self):
    print('Info(Name: {0}, Phon: {1}'.format(self.Name, self.Phone))

  def PrintPersonData(self):
    print('Person(Name: {0}, Phone: {1}'.format(self.Name, self.Phone))
```


```python
class Student(Person): 
  def __init__(self, name, phone, subject, studentID):
    self.Name = name
    self.Phone = phone
    self.Subject = subject
    self.StudentID = studentID
```


```python
p = Person('Karen', '010-3234-3234')
s = Student('Bell', '010-3234-3234', 'Math', '5443234')
```


```python
# __dict__ 
print('Person: ', p.__dict__)   # Show contents in the class as a 'dictionary' type
print('Student: ', s.__dict__)
```

    Person:  {'Name': 'Karen', 'Phone': '010-3234-3234'}
    Student:  {'Name': 'Bell', 'Phone': '010-3234-3234', 'Subject': 'Math', 'StudentID': '5443234'}
    


```python
s.PrintInfo()
```

    Info(Name: Bell, Phon: 010-3234-3234
    


```python
# Check relationships

print(issubclass(Student, Person))  # issubclass(target_subclass, parent_parentclass)
print(issubclass(Person, Student))
```

    True
    False
    


```python
# A child class Student

class Student(Person): 
  def __init__(self, name, phone, subject, studentID):
    self.Name = name
    self.Phone = phone
    self.Subject = subject
    self.StudentID = studentID

    Person.__init__(self, name, phone)

  def PrintStudentData(self):
    print('Student (Name: {0}, Student ID: {1})'.format(self.Name, self.StudentID))

  def PrintPersonData(self):
    # return super().PrintPersonData()   
    print('{0} is majoring in {1}.'.format(self.Name, self.Subject))  # Over Write

  
```


```python
s = Student('Karen', '010-3234-2342', 'Math', '5664323')

#print(s.PrintStudentData())  # the result will show 'None' 
#print(s.PrintPersonData())

s.PrintStudentData()
s.PrintPersonData()
```

    Student (Name: Karen, Student ID: 5664323)
    Karen is majoring in Math.
    

# Excercise 5: Multiple Inheritance


```python
class Tiger:
  def Jump(self):
    print('Tiger!')
```


```python
class Lion:
  def Bite(self):
    print('Lion!!')
```


```python
class Cat(Tiger, Lion): 
  def Play(self):
    print('MEOW!')
```


```python
l = Cat()

l.Jump()
l.Bite()
l.Play()

```

    Tiger!
    Lion!!
    MEOW!
    
