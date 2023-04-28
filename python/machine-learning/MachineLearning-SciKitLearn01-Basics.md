# 1. Preparation

## 1.1. Load Modules


```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt 
```

## 1.2. Get Dataset from SciKit Learn Package


```python
from sklearn.datasets import load_iris
iris = load_iris()
```

## 1.3. Look Up Dataset


```python
print(iris.DESCR)   # Check data info
```

    .. _iris_dataset:
    
    Iris plants dataset
    --------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica
                    
        :Summary Statistics:
    
        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
        ============== ==== ==== ======= ===== ====================
    
        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
    from Fisher's paper. Note that it's the same as in R, but not as in the UCI
    Machine Learning Repository, which has two wrong data points.
    
    This is perhaps the best known database to be found in the
    pattern recognition literature.  Fisher's paper is a classic in the field and
    is referenced frequently to this day.  (See Duda & Hart, for example.)  The
    data set contains 3 classes of 50 instances each, where each class refers to a
    type of iris plant.  One class is linearly separable from the other 2; the
    latter are NOT linearly separable from each other.
    
    .. topic:: References
    
       - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
         Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
         Mathematical Statistics" (John Wiley, NY, 1950).
       - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
         (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
       - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
         Structure and Classification Rule for Recognition in Partially Exposed
         Environments".  IEEE Transactions on Pattern Analysis and Machine
         Intelligence, Vol. PAMI-2, No. 1, 67-71.
       - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
         on Information Theory, May 1972, 431-433.
       - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
         conceptual clustering system finds 3 classes in the data.
       - Many, many more ...
    

## 1.4. Get Data


```python
data = iris.data
label = iris.target
columns = iris.feature_names
```


```python
data  = pd.DataFrame(data, columns=columns)
```

## 1.5. Look Up Data


```python
data.head()
```





  <div id="df-458f3242-5815-4a7c-af34-2264cb12551a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-458f3242-5815-4a7c-af34-2264cb12551a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-458f3242-5815-4a7c-af34-2264cb12551a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-458f3242-5815-4a7c-af34-2264cb12551a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
data.shape
```




    (150, 4)



# 2. Data Splitting: Train & Test Data

## 2.1. Load Modules


```python
from sklearn.model_selection import train_test_split
```

## 2.2. Splitting data to Train set and Test set


```python
# Just split
train_test_split(data, label)
```




    [     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
     17                 5.1               3.5                1.4               0.3
     18                 5.7               3.8                1.7               0.3
     92                 5.8               2.6                4.0               1.2
     142                5.8               2.7                5.1               1.9
     122                7.7               2.8                6.7               2.0
     ..                 ...               ...                ...               ...
     12                 4.8               3.0                1.4               0.1
     33                 5.5               4.2                1.4               0.2
     127                6.1               3.0                4.9               1.8
     108                6.7               2.5                5.8               1.8
     36                 5.5               3.5                1.3               0.2
     
     [112 rows x 4 columns],
          sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
     141                6.9               3.1                5.1               2.3
     79                 5.7               2.6                3.5               1.0
     30                 4.8               3.1                1.6               0.2
     2                  4.7               3.2                1.3               0.2
     27                 5.2               3.5                1.5               0.2
     9                  4.9               3.1                1.5               0.1
     5                  5.4               3.9                1.7               0.4
     34                 4.9               3.1                1.5               0.2
     21                 5.1               3.7                1.5               0.4
     78                 6.0               2.9                4.5               1.5
     105                7.6               3.0                6.6               2.1
     116                6.5               3.0                5.5               1.8
     59                 5.2               2.7                3.9               1.4
     140                6.7               3.1                5.6               2.4
     43                 5.0               3.5                1.6               0.6
     64                 5.6               2.9                3.6               1.3
     149                5.9               3.0                5.1               1.8
     7                  5.0               3.4                1.5               0.2
     26                 5.0               3.4                1.6               0.4
     28                 5.2               3.4                1.4               0.2
     88                 5.6               3.0                4.1               1.3
     62                 6.0               2.2                4.0               1.0
     119                6.0               2.2                5.0               1.5
     69                 5.6               2.5                3.9               1.1
     57                 4.9               2.4                3.3               1.0
     11                 4.8               3.4                1.6               0.2
     48                 5.3               3.7                1.5               0.2
     46                 5.1               3.8                1.6               0.2
     63                 6.1               2.9                4.7               1.4
     73                 6.1               2.8                4.7               1.2
     66                 5.6               3.0                4.5               1.5
     115                6.4               3.2                5.3               2.3
     53                 5.5               2.3                4.0               1.3
     139                6.9               3.1                5.4               2.1
     65                 6.7               3.1                4.4               1.4
     85                 6.0               3.4                4.5               1.6
     58                 6.6               2.9                4.6               1.3
     25                 5.0               3.0                1.6               0.2,
     array([0, 0, 1, 2, 2, 0, 0, 0, 2, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 1, 1, 0,
            2, 2, 0, 1, 1, 0, 0, 0, 1, 2, 2, 0, 1, 0, 0, 1, 2, 1, 2, 1, 2, 2,
            1, 1, 2, 2, 2, 0, 0, 0, 2, 2, 2, 1, 0, 0, 2, 1, 2, 2, 1, 2, 2, 0,
            2, 0, 0, 2, 1, 1, 2, 2, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 2,
            2, 1, 1, 2, 1, 2, 0, 0, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 0, 0, 0, 2,
            2, 0]),
     array([2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 0, 1, 2, 0, 0, 0, 1, 1,
            2, 1, 1, 0, 0, 0, 1, 1, 1, 2, 1, 2, 1, 1, 1, 0])]




```python
# Set a ratio of train:data
train_test_split(data, label, test_size=0.2)
```




    [     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
     40                 5.0               3.5                1.3               0.3
     132                6.4               2.8                5.6               2.2
     8                  4.4               2.9                1.4               0.2
     2                  4.7               3.2                1.3               0.2
     149                5.9               3.0                5.1               1.8
     ..                 ...               ...                ...               ...
     86                 6.7               3.1                4.7               1.5
     62                 6.0               2.2                4.0               1.0
     53                 5.5               2.3                4.0               1.3
     34                 4.9               3.1                1.5               0.2
     146                6.3               2.5                5.0               1.9
     
     [120 rows x 4 columns],
          sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
     75                 6.6               3.0                4.4               1.4
     10                 5.4               3.7                1.5               0.2
     5                  5.4               3.9                1.7               0.4
     70                 5.9               3.2                4.8               1.8
     71                 6.1               2.8                4.0               1.3
     67                 5.8               2.7                4.1               1.0
     109                7.2               3.6                6.1               2.5
     138                6.0               3.0                4.8               1.8
     27                 5.2               3.5                1.5               0.2
     110                6.5               3.2                5.1               2.0
     141                6.9               3.1                5.1               2.3
     14                 5.8               4.0                1.2               0.2
     140                6.7               3.1                5.6               2.4
     95                 5.7               3.0                4.2               1.2
     72                 6.3               2.5                4.9               1.5
     83                 6.0               2.7                5.1               1.6
     97                 6.2               2.9                4.3               1.3
     11                 4.8               3.4                1.6               0.2
     42                 4.4               3.2                1.3               0.2
     129                7.2               3.0                5.8               1.6
     103                6.3               2.9                5.6               1.8
     38                 4.4               3.0                1.3               0.2
     114                5.8               2.8                5.1               2.4
     84                 5.4               3.0                4.5               1.5
     81                 5.5               2.4                3.7               1.0
     6                  4.6               3.4                1.4               0.3
     131                7.9               3.8                6.4               2.0
     52                 6.9               3.1                4.9               1.5
     4                  5.0               3.6                1.4               0.2
     36                 5.5               3.5                1.3               0.2,
     array([0, 2, 0, 0, 2, 1, 1, 0, 2, 0, 2, 1, 0, 2, 2, 0, 2, 0, 1, 2, 1, 1,
            2, 1, 1, 2, 2, 2, 1, 2, 0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 2, 1, 1, 0,
            2, 0, 1, 1, 1, 0, 0, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 0, 0, 1,
            1, 0, 1, 2, 0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 1, 2, 0, 2, 0, 0, 2, 0,
            2, 0, 0, 0, 0, 2, 2, 1, 1, 0, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2,
            1, 0, 0, 0, 0, 1, 1, 1, 0, 2]),
     array([1, 0, 0, 1, 1, 1, 2, 2, 0, 2, 2, 0, 2, 1, 1, 1, 1, 0, 0, 2, 2, 0,
            2, 1, 1, 0, 2, 1, 0, 0])]




```python
# Randomly mix data
train_test_split(data, label, test_size=0.2, random_state=2023)
```




    [     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
     9                  4.9               3.1                1.5               0.1
     38                 4.4               3.0                1.3               0.2
     18                 5.7               3.8                1.7               0.3
     119                6.0               2.2                5.0               1.5
     98                 5.1               2.5                3.0               1.1
     ..                 ...               ...                ...               ...
     52                 6.9               3.1                4.9               1.5
     116                6.5               3.0                5.5               1.8
     3                  4.6               3.1                1.5               0.2
     25                 5.0               3.0                1.6               0.2
     87                 6.3               2.3                4.4               1.3
     
     [120 rows x 4 columns],
          sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
     128                6.4               2.8                5.6               2.1
     74                 6.4               2.9                4.3               1.3
     99                 5.7               2.8                4.1               1.3
     107                7.3               2.9                6.3               1.8
     76                 6.8               2.8                4.8               1.4
     113                5.7               2.5                5.0               2.0
     84                 5.4               3.0                4.5               1.5
     96                 5.7               2.9                4.2               1.3
     4                  5.0               3.6                1.4               0.2
     62                 6.0               2.2                4.0               1.0
     30                 4.8               3.1                1.6               0.2
     86                 6.7               3.1                4.7               1.5
     41                 4.5               2.3                1.3               0.3
     137                6.4               3.1                5.5               1.8
     17                 5.1               3.5                1.4               0.3
     120                6.9               3.2                5.7               2.3
     32                 5.2               4.1                1.5               0.1
     57                 4.9               2.4                3.3               1.0
     35                 5.0               3.2                1.2               0.2
     16                 5.4               3.9                1.3               0.4
     94                 5.6               2.7                4.2               1.3
     2                  4.7               3.2                1.3               0.2
     122                7.7               2.8                6.7               2.0
     50                 7.0               3.2                4.7               1.4
     23                 5.1               3.3                1.7               0.5
     14                 5.8               4.0                1.2               0.2
     21                 5.1               3.7                1.5               0.4
     135                7.7               3.0                6.1               2.3
     81                 5.5               2.4                3.7               1.0
     48                 5.3               3.7                1.5               0.2,
     array([0, 0, 0, 2, 1, 2, 0, 1, 2, 0, 1, 2, 1, 1, 1, 2, 1, 1, 1, 0, 0, 2,
            0, 0, 1, 2, 1, 1, 0, 0, 1, 2, 2, 0, 1, 2, 1, 1, 1, 2, 0, 1, 2, 1,
            2, 2, 1, 0, 1, 1, 0, 2, 2, 0, 1, 1, 0, 2, 0, 1, 1, 2, 0, 0, 0, 2,
            2, 0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 2, 0,
            0, 2, 2, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0, 0, 1, 2, 0, 1,
            2, 0, 1, 0, 2, 1, 2, 0, 0, 1]),
     array([2, 1, 1, 2, 1, 2, 1, 1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 1, 0, 0, 1, 0,
            2, 1, 0, 0, 0, 2, 1, 0])]




```python
# Load Data
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=2023)
```


```python
y_train
```




    array([0, 0, 0, 2, 1, 2, 0, 1, 2, 0, 1, 2, 1, 1, 1, 2, 1, 1, 1, 0, 0, 2,
           0, 0, 1, 2, 1, 1, 0, 0, 1, 2, 2, 0, 1, 2, 1, 1, 1, 2, 0, 1, 2, 1,
           2, 2, 1, 0, 1, 1, 0, 2, 2, 0, 1, 1, 0, 2, 0, 1, 1, 2, 0, 0, 0, 2,
           2, 0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 2, 0,
           0, 2, 2, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0, 0, 1, 2, 0, 1,
           2, 0, 1, 0, 2, 1, 2, 0, 0, 1])


