# 1. Preparation

## 1.1. Installation

### Install ODBC Driver on Windows


### Install pyodbc in Python


```python
!pip install pyodbc
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting pyodbc
      Downloading pyodbc-4.0.39-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (343 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m343.5/343.5 kB[0m [31m14.1 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: pyodbc
    Successfully installed pyodbc-4.0.39
    

## 1.2. Make a connection to Azure DB Server


```python
server = 'mysqlserver-meow94.database.windows.net'
database = 'BikeStores'
username = 'meow94'
password = 'qwer1234!@#$'
driver = '{ODBC Driver 18 for SQL Server}'

cnxn = pyodbc.connect('DRIVER='+driver+';PORT=1433;SERVER='+server+';PORT=1443;DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()
```
