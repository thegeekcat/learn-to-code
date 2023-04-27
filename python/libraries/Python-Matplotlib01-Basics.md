# 1. Preparation


```python
# Load 'matplotlib'

import matplotlib as mpl
import matplotlib.pyplot as plt   
```


```python
# Set a graph style as 'classic'
plt.style.use('classic') 
```


```python
# Load 'numpy' for data

import numpy as np
x = np.linspace(0, 10, 100)  # np.linspace(Start, End, Num of generations) -> Generate 100 values between 0 and 100
x
```




    array([ 0.        ,  0.1010101 ,  0.2020202 ,  0.3030303 ,  0.4040404 ,
            0.50505051,  0.60606061,  0.70707071,  0.80808081,  0.90909091,
            1.01010101,  1.11111111,  1.21212121,  1.31313131,  1.41414141,
            1.51515152,  1.61616162,  1.71717172,  1.81818182,  1.91919192,
            2.02020202,  2.12121212,  2.22222222,  2.32323232,  2.42424242,
            2.52525253,  2.62626263,  2.72727273,  2.82828283,  2.92929293,
            3.03030303,  3.13131313,  3.23232323,  3.33333333,  3.43434343,
            3.53535354,  3.63636364,  3.73737374,  3.83838384,  3.93939394,
            4.04040404,  4.14141414,  4.24242424,  4.34343434,  4.44444444,
            4.54545455,  4.64646465,  4.74747475,  4.84848485,  4.94949495,
            5.05050505,  5.15151515,  5.25252525,  5.35353535,  5.45454545,
            5.55555556,  5.65656566,  5.75757576,  5.85858586,  5.95959596,
            6.06060606,  6.16161616,  6.26262626,  6.36363636,  6.46464646,
            6.56565657,  6.66666667,  6.76767677,  6.86868687,  6.96969697,
            7.07070707,  7.17171717,  7.27272727,  7.37373737,  7.47474747,
            7.57575758,  7.67676768,  7.77777778,  7.87878788,  7.97979798,
            8.08080808,  8.18181818,  8.28282828,  8.38383838,  8.48484848,
            8.58585859,  8.68686869,  8.78787879,  8.88888889,  8.98989899,
            9.09090909,  9.19191919,  9.29292929,  9.39393939,  9.49494949,
            9.5959596 ,  9.6969697 ,  9.7979798 ,  9.8989899 , 10.        ])




```python
# Sin: Starting from '0'
# Cos: Starting from '1'
# => Same graph but different starting point
```


```python
print(x) 
print(np.sin(x))
```

    [ 0.          0.1010101   0.2020202   0.3030303   0.4040404   0.50505051
      0.60606061  0.70707071  0.80808081  0.90909091  1.01010101  1.11111111
      1.21212121  1.31313131  1.41414141  1.51515152  1.61616162  1.71717172
      1.81818182  1.91919192  2.02020202  2.12121212  2.22222222  2.32323232
      2.42424242  2.52525253  2.62626263  2.72727273  2.82828283  2.92929293
      3.03030303  3.13131313  3.23232323  3.33333333  3.43434343  3.53535354
      3.63636364  3.73737374  3.83838384  3.93939394  4.04040404  4.14141414
      4.24242424  4.34343434  4.44444444  4.54545455  4.64646465  4.74747475
      4.84848485  4.94949495  5.05050505  5.15151515  5.25252525  5.35353535
      5.45454545  5.55555556  5.65656566  5.75757576  5.85858586  5.95959596
      6.06060606  6.16161616  6.26262626  6.36363636  6.46464646  6.56565657
      6.66666667  6.76767677  6.86868687  6.96969697  7.07070707  7.17171717
      7.27272727  7.37373737  7.47474747  7.57575758  7.67676768  7.77777778
      7.87878788  7.97979798  8.08080808  8.18181818  8.28282828  8.38383838
      8.48484848  8.58585859  8.68686869  8.78787879  8.88888889  8.98989899
      9.09090909  9.19191919  9.29292929  9.39393939  9.49494949  9.5959596
      9.6969697   9.7979798   9.8989899  10.        ]
    [ 0.          0.10083842  0.20064886  0.2984138   0.39313661  0.48385164
      0.56963411  0.64960951  0.72296256  0.78894546  0.84688556  0.8961922
      0.93636273  0.96698762  0.98775469  0.99845223  0.99897117  0.98930624
      0.96955595  0.93992165  0.90070545  0.85230712  0.79522006  0.73002623
      0.65739025  0.57805259  0.49282204  0.40256749  0.30820902  0.21070855
      0.11106004  0.01027934 -0.09060615 -0.19056796 -0.28858706 -0.38366419
     -0.47483011 -0.56115544 -0.64176014 -0.7158225  -0.7825875  -0.84137452
     -0.89158426 -0.93270486 -0.96431712 -0.98609877 -0.99782778 -0.99938456
     -0.99075324 -0.97202182 -0.94338126 -0.90512352 -0.85763861 -0.80141062
     -0.73701276 -0.66510151 -0.58640998 -0.50174037 -0.41195583 -0.31797166
     -0.22074597 -0.12126992 -0.0205576   0.0803643   0.18046693  0.27872982
      0.37415123  0.46575841  0.55261747  0.63384295  0.7086068   0.77614685
      0.83577457  0.8868821   0.92894843  0.96154471  0.98433866  0.99709789
      0.99969234  0.99209556  0.97438499  0.94674118  0.90944594  0.86287948
      0.8075165   0.74392141  0.6727425   0.59470541  0.51060568  0.42130064
      0.32770071  0.23076008  0.13146699  0.03083368 -0.07011396 -0.17034683
     -0.26884313 -0.36459873 -0.45663749 -0.54402111]


# 2. Basics

## 2.1. 'sin' and 'cos'


```python
# x-axis: simple increase from 0 to 10
# y = sin(x): Repeat increasing and decreasing from '0' to '-1' and '1'
plt.plot(x, np.sin(x))
```




    [<matplotlib.lines.Line2D at 0x7ff058368eb0>]

![Python-Matplotlib01-Basics-01]({{site.url}}/libraries/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img01.png)



```python
# x: 0부터  10까지 단순 증가
# y = cos(x): 1부터시작해서 1과 -1 사이 증감 반복
plt.plot(x, np.cos(x))
```




    [<matplotlib.lines.Line2D at 0x7ff058272130>]

![Python-Matplotlib01-Basics-02]({{site.url}}/assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img02.png)



```python
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.plot(np.sin(x), np.cos(x))
```




    [<matplotlib.lines.Line2D at 0x7ff0581dafa0>]

![Python-Matplotlib01-Basics-03]({{site.url}}/assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img03.png)





## 2.2. Create a Canvas and Save it as a 'png' format


```python
# Create a canvas(=figure)
fig = plt.figure()  

# Draw graphs
plt.plot(x, np.sin(x), '-')  # 실선 표현
plt.plot(x, np.cos(x), '--')  # dashline 표현
plt.plot(x, np.cos(x+1), '.')  # dotline 표현
plt.plot(x, np.cos(x-1), '-.')  # dash-dotline 표현

# Save the graphs as 'my_figure.png'
fig.savefig('my_figure.png') # 그래프 저장 시 png 포맷이 선명도 제일 좋음
```

![Python-Matplotlib01-Basics-04]({{site.url}}/assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img04.png)    



```python
# Load the image
from IPython.display import Image
Image('my_figure.png')
```

![Python-Matplotlib01-Basics-05]({{site.url}}/assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img05.png)




```python
# Check supported image file types
fig.canvas.get_supported_filetypes()
```




    {'eps': 'Encapsulated Postscript',
     'jpg': 'Joint Photographic Experts Group',
     'jpeg': 'Joint Photographic Experts Group',
     'pdf': 'Portable Document Format',
     'pgf': 'PGF code for LaTeX',
     'png': 'Portable Network Graphics',
     'ps': 'Postscript',
     'raw': 'Raw RGBA bitmap',
     'rgba': 'Raw RGBA bitmap',
     'svg': 'Scalable Vector Graphics',
     'svgz': 'Scalable Vector Graphics',
     'tif': 'Tagged Image File Format',
     'tiff': 'Tagged Image File Format',
     'webp': 'WebP Image Format'}



## 2.3. Draw Two Graphs in A Figure


```python
# Create a 'figure'
plt.figure()
```




    <Figure size 640x480 with 0 Axes>




    <Figure size 640x480 with 0 Axes>



```python
# Create two sub-figures in the figure
plt.subplot(2, 1, 1)   # plt.subplot(rows, cols, index(panel number))
plt.subplot(2, 1, 2)   
```




    <Axes: >

![Python-Matplotlib01-Basics-06](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img06.png)



```python
# Create 3 sub-figures in a figure
plt.subplot(3, 3, 1)   # plt.subplot(rows, cols, panel number)
plt.subplot(3, 3, 5)  
plt.subplot(3, 3, 9)  
```




    <Axes: >

![Python-Matplotlib01-Basics-07](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img07.png)



```python
# Draw two graphs in the two sub-figures
plt.subplot(2, 1, 1)   # plt.subplot(rows, cols, panel number)
plt.plot(x, np.sin(x))
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))   
```




    [<matplotlib.lines.Line2D at 0x7ff038ea94f0>]

![Python-Matplotlib01-Basics-08](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img08.png)


  


## 2.4. Line Colors


```python
plt.plot(x, np.sin(x), color='blue')
plt.plot(x, np.sin(x-1), color='k') # k = black, b = blue
plt.plot(x, np.sin(x-2), color='0.75')  # 회색조 농도 표현 -> 75% 회색
plt.plot(x, np.sin(x-3), color='#663300')
plt.plot(x, np.sin(x-4), color=(1.0, 0.2, 0.3))  # RGB 표현 방식
plt.plot(x, np.sin(x-5), color='chartreuse')
#plt.plot(x, np.cos(x))
#plt.plot(x, np.cos(x-1))
#plt.plot(x, np.cos(x-2))
#plt.plot(x, np.cos(x-3))
#plt.plot(x, np.cos(x-4))
#plt.plot(x, np.cos(x-5))

```




    [<matplotlib.lines.Line2D at 0x7ff038dc6910>]

![Python-Matplotlib01-Basics-09](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img09.png)





## 2.5. Line Styles


```python
# Dash types by 'name'

plt.plot(x, x+0, linestyle='solid')    # solid line
plt.plot(x, x+1, linestyle='dashed')   # dash line
plt.plot(x, x+2, linestyle='dashdot')  # dash-dot line
plt.plot(x, x+3, linestyle='dotted')   # dot line
```




    [<matplotlib.lines.Line2D at 0x7ff038db5eb0>]

![Python-Matplotlib01-Basics-10](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img10.png)





```python
# Dash types by 'symbol'

plt.plot(x, x+0, linestyle='-')    # solid line
plt.plot(x, x+1, linestyle='--')   # dash line
plt.plot(x, x+2, linestyle='-.')   # dash-dot line
plt.plot(x, x+3, linestyle=':')    # dot line
```




    [<matplotlib.lines.Line2D at 0x7ff038cc0a30>]

![Python-Matplotlib01-Basics-11](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img11.png)



```python
# Set dash types and colors
plt.plot(x, x+0, '-g')   # solid green
plt.plot(x, x+1, '--b')  # dashed blue
plt.plot(x, x+2, '-.y')  # dash-dot yellow
plt.plot(x, x+3, ':r')   # dotted red
```




    [<matplotlib.lines.Line2D at 0x7ff0581f4280>]

![Python-Matplotlib01-Basics-12](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img12.png)


 


## 2.6. Adjust Graph Positioning


```python
# Adjust Positioning - One axis
plt.plot(x, np.sin(x))
plt.xlim(3, 11)  # limite: xlim(Start, End)
plt.ylim(-1.5, 1.5)
```




    (-1.5, 1.5)

![Python-Matplotlib01-Basics-13](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img13.png)



```python
# Adjust Positioning: Two Axises
plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5])  # plt.axis([x-Start, x-End, y-Start, y-End])
```




    (-1.0, 11.0, -1.5, 1.5)

![Python-Matplotlib01-Basics-14](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img14.png)





## 2.7. Set a Size of Figures depending on X-axis and Y-axis


```python
# Figure: 'tight'

plt.plot(x, np.sin(x))
plt.axis('tight')
```




    (0.0, 10.0, -0.9993845576124357, 0.9996923408861117)

![Python-Matplotlib01-Basics-15](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img15.png)



```python
# Figure: 'equal' -> The same ratio of x:y
plt.plot(x, np.sin(x))
plt.axis('equal')  
```




    (0.0, 10.0, -1.0, 1.0)

![Python-Matplotlib01-Basics-16](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img16.png)





## 2.8. Title and Labels


```python
# Add Labels and Title
plt.plot(x, np.sin(x))
plt.title('A Sine Curve')  # Title
plt.xlabel('x')   # X-axis label
plt.ylabel('sin(x)')  # Y-axis label
```




    Text(0, 0.5, 'sin(x)')

![Python-Matplotlib01-Basics-17](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img17.png)



```python
plt.plot(x, np.sin(x), '-g')
plt.plot(x, np.cos(x), ':b')
plt.axis('equal')
```




    (0.0, 10.0, -1.0, 1.0)

![Python-Matplotlib01-Basics-18](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img18.png)


​    


## 2.9. Excercise


```python
plt.title('A Graph')
plt.plot(x, np.sin(x), '-g', label='sin(x)')  # plt.plot(x-axis, y-axis, Line style, label)
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')  # a size of feature
plt.legend # Add legends
plt.xlabel('X-axis') # Add label of x-axis
plt.ylabel('Y-axis') # Add label of y-axis
```




    Text(0, 0.5, 'Y-axis')

![Python-Matplotlib01-Basics-19](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img19.png)





# 3. Scatter Plot

## 3.1. Plot() Function


- Plat
  - Draw points(markers) in a diagram


```python
x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.plot(x, y, 'o', color='black')   # plt.plot(x-axis, y-axis, marker)
```




    [<matplotlib.lines.Line2D at 0x7ff0389b81c0>]

![Python-Matplotlib01-Basics-20](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img20.png)



```python
# np.random.RandomState(seed value) -> seed value: a reference to generate random values
plt.style.use('bmh')  # 'bmh' = beysian style
rng = np.random.RandomState(0)  # Generate random values based on '0'
for marker in ['o', 'p', ',', 'b', 'd', 'x', '+', 'v', '^', '<', '>', '1', '2', '3', '4']:
  plt.plot(rng.rand(5), rng.rand(5), marker, label='marker: {0}'.format(marker))
  plt.legend(numpoints=1)  # Display a 1-gap grid
  plt.xlim(0, 1.8) # Increase a range of x-axis
```

​    ![Python-Matplotlib01-Basics-21](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img21.png)





```python
# Plot + Graph
plt.plot(x, y, '--o', color='b')  
```




    [<matplotlib.lines.Line2D at 0x7ff0388407f0>]

![Python-Matplotlib01-Basics-22](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img22.png)



```python
plt.style.use('bmh')  
plt.plot(x, y, '-o', 
         markersize=7,  # Marker size
         linewidth=3,   # Width - Line 
         markeredgewidth=3,  # Width - Marker edge
         color='red',        # Color - line
         markerfacecolor='white',  # Color - Marker face
         markeredgecolor='blue')   # Color - Marker edge

```




    [<matplotlib.lines.Line2D at 0x7ff038772b20>]

![Python-Matplotlib01-Basics-23](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img23.png)





## 3.2. Scatter() Function

- Scatter() Function
 - Draw scatter plots


```python
# Random Value: rng.randint 

#x = rng.randint(10)  # random int: A random value between 0 and 9
#x = rng.randint(1, 10)  # random int: A random value between 1 and 10

#x = rng.rand(100) # Positive numbers only
#x = rng.randn(100) # Random Negative: Positive and Negative
```


```python
# Dice 10 times

for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Number of dice turns
  print(i, ": ", np.random.randint(1, 7))  # Dice scale from 1 to 6
```

    1 :  6
    2 :  3
    3 :  5
    4 :  6
    5 :  6
    6 :  6
    7 :  2
    8 :  4
    9 :  4
    10 :  3



```python
# Draw a basic 'Scatter Plot'
plt.style.use('classic')

rng = np.random.RandomState(0)  # rng = random number generator
x = rng.randn(100)
y = rng.randn(100)

plt.scatter(x, y)
```




    <matplotlib.collections.PathCollection at 0x7ff0386e33a0>

![Python-Matplotlib01-Basics-24](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img24.png)



```python
# Set colors to plots
color = rng.rand(100)

plt.scatter(x, y, c=color, alpha=0.3)  # alpha:  transparency -> 0.3 = 30%
plt.scatter(x, y, c=color, alpha=0.3, cmap='viridis') # cmap: color map
```




    <matplotlib.collections.PathCollection at 0x7ff0386702e0>

![Python-Matplotlib01-Basics-25](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img25.png)



```python
# Set a size of plots
sizes = rng.rand(100) * 1000
plt.scatter(x, y, c=color, alpha=0.3, s=sizes)  
```




    <matplotlib.collections.PathCollection at 0x7ff0385def70>

![Python-Matplotlib01-Basics-26](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img26.png)





## 3.3. Scatter Plot example: Iris dataset 


```python
# Load the 'iris' Dataset from 'sklearn'
from sklearn.datasets import load_iris

iris = load_iris()
features = iris.data.T
```


```python
# Draw a basic scatter plot

plt.scatter(features[2], features[3])
```




    <matplotlib.collections.PathCollection at 0x7ff02cc47430>

![Python-Matplotlib01-Basics-27](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img27.png)



```python
# Set colors

plt.scatter(features[2], features[3], c=iris.target, alpha=0.4, cmap='viridis')
```




    <matplotlib.collections.PathCollection at 0x7ff02cb869d0>

![Python-Matplotlib01-Basics-28](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img28.png)


 

```python
# Set a size of plots
plt.scatter(features[2], features[3], c=iris.target, alpha=0.3, cmap='viridis', s=features[3]*100)
```




    <matplotlib.collections.PathCollection at 0x7ff02cb0dc10>

![Python-Matplotlib01-Basics-29](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img29.png)



```python
# Set 'Label' names

plt.scatter(features[2], features[3], c=iris.target, alpha=0.3, cmap='viridis', s=features[3]*100)

plt.title('Dataset')
plt.legend()
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
iris.feature_names[3]
```

    WARNING:matplotlib.legend:No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.





    'petal width (cm)'

![Python-Matplotlib01-Basics-30](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img30.png)





# 4. Error Bar


```python
x = np.linspace(0, 10, 50)  # np.linspace(Start, End, Number of values)
x
```




    array([ 0.        ,  0.20408163,  0.40816327,  0.6122449 ,  0.81632653,
            1.02040816,  1.2244898 ,  1.42857143,  1.63265306,  1.83673469,
            2.04081633,  2.24489796,  2.44897959,  2.65306122,  2.85714286,
            3.06122449,  3.26530612,  3.46938776,  3.67346939,  3.87755102,
            4.08163265,  4.28571429,  4.48979592,  4.69387755,  4.89795918,
            5.10204082,  5.30612245,  5.51020408,  5.71428571,  5.91836735,
            6.12244898,  6.32653061,  6.53061224,  6.73469388,  6.93877551,
            7.14285714,  7.34693878,  7.55102041,  7.75510204,  7.95918367,
            8.16326531,  8.36734694,  8.57142857,  8.7755102 ,  8.97959184,
            9.18367347,  9.3877551 ,  9.59183673,  9.79591837, 10.        ])




```python

y = np.sin(x)

plt.scatter(x, y)
```




    <matplotlib.collections.PathCollection at 0x7ff02c9b5ee0>

![Python-Matplotlib01-Basics-31](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img31.png)


 

```python
# The scatter plot above is too constant -> Randomize y-axis

y = np.sin(x + np.random.randn(50)) # randn(50): random values between -50 and 50 (* randn = random negative)

plt.scatter(x, y)
```




    <matplotlib.collections.PathCollection at 0x7ff02c934a30>

![Python-Matplotlib01-Basics-32](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img32.png)



```python
# Error Bar
plt.errorbar(x, y)
```




    <ErrorbarContainer object of 3 artists>

![Python-Matplotlib01-Basics-33](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img33.png)



```python
# Set formats to Error Bar
plt.errorbar(x, y, fmt='.k')  # fmt = format,   '.k': dot+black
```




    <ErrorbarContainer object of 3 artists>

![Python-Matplotlib01-Basics-34](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img34.png)



```python
# Define a range of Errors

dy = 0.8  # Range of errors
plt.errorbar(x, y, yerr=dy, fmt='.k')  # err=dy -> Mark the erros on y-axis -> yerr=dy
```




    <ErrorbarContainer object of 3 artists>

![Python-Matplotlib01-Basics-35](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img35.png)



```python
# Set styles - colors
plt.errorbar(x, y, yerr=dy, fmt='o', color='k')
```




    <ErrorbarContainer object of 3 artists>

![Python-Matplotlib01-Basics-36](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img36.png)



```python
# Set styles - error bar
plt.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgrey')  # ecolor: error color
```




    <ErrorbarContainer object of 3 artists>

![Python-Matplotlib01-Basics-37](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img37.png)



```python
# Set styles - Set a width of error lines
plt.title('Error Bar')
plt.errorbar(x, y, yerr=dy, fmt='o', color='k', ecolor='lightgrey', elinewidth=3) 
```




    <ErrorbarContainer object of 3 artists>

![Python-Matplotlib01-Basics-38](../assets/images/Python-Matplotlib01-Basics/Python-Matplotlib01-Basics-img38.png)



```python

```
