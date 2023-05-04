# 1. Preparation

## 1.1. Basic Settings


```python
# Import modules 
from azureml.core import Workspace
```


```python
# Define stauts of the current settings
ws = Workspace.from_config()
```


```python
# Check information
print('Workspace Same: ' + ws.name,
        '\nAzure Region: ' + ws.location,
        '\nSubscription ID: ' + ws.subscription_id,
        '\nResource Group: ' + ws.resource_group)
```

    Workspace Same: labmeow94ml 
    Azure Region: australiaeast 
    Subscription ID: 27db5ec6-d206-4028-b5e1-6004dca5eeef 
    Resource Group: rg94
    

## 1.2. Settings for Experiment


```python
# Prepare 

from azureml.core import Experiment
```


```python
# Define Experiment
experiment = Experiment(workspace=ws,
                        name='diabetes-experiment')
```

## 1.3. Prepare Dataset


```python
# Import modules
from azureml.opendatasets import Diabetes
from sklearn.model_selection import train_test_split
```


```python
# Check the original dataset
Diabetes.get_tabular_dataset().to_pandas_dataframe()
# to_pandas_dataframe(): Bring data as a type of Pandas DataFrame
```




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
      <th>AGE</th>
      <th>SEX</th>
      <th>BMI</th>
      <th>BP</th>
      <th>S1</th>
      <th>S2</th>
      <th>S3</th>
      <th>S4</th>
      <th>S5</th>
      <th>S6</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>2</td>
      <td>32.1</td>
      <td>101.00</td>
      <td>157</td>
      <td>93.2</td>
      <td>38.0</td>
      <td>4.00</td>
      <td>4.8598</td>
      <td>87</td>
      <td>151</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48</td>
      <td>1</td>
      <td>21.6</td>
      <td>87.00</td>
      <td>183</td>
      <td>103.2</td>
      <td>70.0</td>
      <td>3.00</td>
      <td>3.8918</td>
      <td>69</td>
      <td>75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72</td>
      <td>2</td>
      <td>30.5</td>
      <td>93.00</td>
      <td>156</td>
      <td>93.6</td>
      <td>41.0</td>
      <td>4.00</td>
      <td>4.6728</td>
      <td>85</td>
      <td>141</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24</td>
      <td>1</td>
      <td>25.3</td>
      <td>84.00</td>
      <td>198</td>
      <td>131.4</td>
      <td>40.0</td>
      <td>5.00</td>
      <td>4.8903</td>
      <td>89</td>
      <td>206</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>1</td>
      <td>23.0</td>
      <td>101.00</td>
      <td>192</td>
      <td>125.4</td>
      <td>52.0</td>
      <td>4.00</td>
      <td>4.2905</td>
      <td>80</td>
      <td>135</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>437</th>
      <td>60</td>
      <td>2</td>
      <td>28.2</td>
      <td>112.00</td>
      <td>185</td>
      <td>113.8</td>
      <td>42.0</td>
      <td>4.00</td>
      <td>4.9836</td>
      <td>93</td>
      <td>178</td>
    </tr>
    <tr>
      <th>438</th>
      <td>47</td>
      <td>2</td>
      <td>24.9</td>
      <td>75.00</td>
      <td>225</td>
      <td>166.0</td>
      <td>42.0</td>
      <td>5.00</td>
      <td>4.4427</td>
      <td>102</td>
      <td>104</td>
    </tr>
    <tr>
      <th>439</th>
      <td>60</td>
      <td>2</td>
      <td>24.9</td>
      <td>99.67</td>
      <td>162</td>
      <td>106.6</td>
      <td>43.0</td>
      <td>3.77</td>
      <td>4.1271</td>
      <td>95</td>
      <td>132</td>
    </tr>
    <tr>
      <th>440</th>
      <td>36</td>
      <td>1</td>
      <td>30.0</td>
      <td>95.00</td>
      <td>201</td>
      <td>125.2</td>
      <td>42.0</td>
      <td>4.79</td>
      <td>5.1299</td>
      <td>85</td>
      <td>220</td>
    </tr>
    <tr>
      <th>441</th>
      <td>36</td>
      <td>1</td>
      <td>19.6</td>
      <td>71.00</td>
      <td>250</td>
      <td>133.2</td>
      <td>97.0</td>
      <td>3.00</td>
      <td>4.5951</td>
      <td>92</td>
      <td>57</td>
    </tr>
  </tbody>
</table>
<p>442 rows × 11 columns</p>
</div>



# 2. Data Preparation


```python
# Create X-axis

x_df = Diabetes.get_tabular_dataset().to_pandas_dataframe().dropna()
# dropna(): Drop 'None' values

x_df
```




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
      <th>AGE</th>
      <th>SEX</th>
      <th>BMI</th>
      <th>BP</th>
      <th>S1</th>
      <th>S2</th>
      <th>S3</th>
      <th>S4</th>
      <th>S5</th>
      <th>S6</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>2</td>
      <td>32.1</td>
      <td>101.00</td>
      <td>157</td>
      <td>93.2</td>
      <td>38.0</td>
      <td>4.00</td>
      <td>4.8598</td>
      <td>87</td>
      <td>151</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48</td>
      <td>1</td>
      <td>21.6</td>
      <td>87.00</td>
      <td>183</td>
      <td>103.2</td>
      <td>70.0</td>
      <td>3.00</td>
      <td>3.8918</td>
      <td>69</td>
      <td>75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72</td>
      <td>2</td>
      <td>30.5</td>
      <td>93.00</td>
      <td>156</td>
      <td>93.6</td>
      <td>41.0</td>
      <td>4.00</td>
      <td>4.6728</td>
      <td>85</td>
      <td>141</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24</td>
      <td>1</td>
      <td>25.3</td>
      <td>84.00</td>
      <td>198</td>
      <td>131.4</td>
      <td>40.0</td>
      <td>5.00</td>
      <td>4.8903</td>
      <td>89</td>
      <td>206</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>1</td>
      <td>23.0</td>
      <td>101.00</td>
      <td>192</td>
      <td>125.4</td>
      <td>52.0</td>
      <td>4.00</td>
      <td>4.2905</td>
      <td>80</td>
      <td>135</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>437</th>
      <td>60</td>
      <td>2</td>
      <td>28.2</td>
      <td>112.00</td>
      <td>185</td>
      <td>113.8</td>
      <td>42.0</td>
      <td>4.00</td>
      <td>4.9836</td>
      <td>93</td>
      <td>178</td>
    </tr>
    <tr>
      <th>438</th>
      <td>47</td>
      <td>2</td>
      <td>24.9</td>
      <td>75.00</td>
      <td>225</td>
      <td>166.0</td>
      <td>42.0</td>
      <td>5.00</td>
      <td>4.4427</td>
      <td>102</td>
      <td>104</td>
    </tr>
    <tr>
      <th>439</th>
      <td>60</td>
      <td>2</td>
      <td>24.9</td>
      <td>99.67</td>
      <td>162</td>
      <td>106.6</td>
      <td>43.0</td>
      <td>3.77</td>
      <td>4.1271</td>
      <td>95</td>
      <td>132</td>
    </tr>
    <tr>
      <th>440</th>
      <td>36</td>
      <td>1</td>
      <td>30.0</td>
      <td>95.00</td>
      <td>201</td>
      <td>125.2</td>
      <td>42.0</td>
      <td>4.79</td>
      <td>5.1299</td>
      <td>85</td>
      <td>220</td>
    </tr>
    <tr>
      <th>441</th>
      <td>36</td>
      <td>1</td>
      <td>19.6</td>
      <td>71.00</td>
      <td>250</td>
      <td>133.2</td>
      <td>97.0</td>
      <td>3.00</td>
      <td>4.5951</td>
      <td>92</td>
      <td>57</td>
    </tr>
  </tbody>
</table>
<p>442 rows × 11 columns</p>
</div>




```python
# Create Y-axis as a Label
y_df = x_df.pop('Y')
y_df
```




    0      151
    1       75
    2      141
    3      206
    4      135
          ... 
    437    178
    438    104
    439    132
    440    220
    441     57
    Name: Y, Length: 442, dtype: int64




```python
# Check X-Axis again after Popping the label
x_df
```




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
      <th>AGE</th>
      <th>SEX</th>
      <th>BMI</th>
      <th>BP</th>
      <th>S1</th>
      <th>S2</th>
      <th>S3</th>
      <th>S4</th>
      <th>S5</th>
      <th>S6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>2</td>
      <td>32.1</td>
      <td>101.00</td>
      <td>157</td>
      <td>93.2</td>
      <td>38.0</td>
      <td>4.00</td>
      <td>4.8598</td>
      <td>87</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48</td>
      <td>1</td>
      <td>21.6</td>
      <td>87.00</td>
      <td>183</td>
      <td>103.2</td>
      <td>70.0</td>
      <td>3.00</td>
      <td>3.8918</td>
      <td>69</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72</td>
      <td>2</td>
      <td>30.5</td>
      <td>93.00</td>
      <td>156</td>
      <td>93.6</td>
      <td>41.0</td>
      <td>4.00</td>
      <td>4.6728</td>
      <td>85</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24</td>
      <td>1</td>
      <td>25.3</td>
      <td>84.00</td>
      <td>198</td>
      <td>131.4</td>
      <td>40.0</td>
      <td>5.00</td>
      <td>4.8903</td>
      <td>89</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>1</td>
      <td>23.0</td>
      <td>101.00</td>
      <td>192</td>
      <td>125.4</td>
      <td>52.0</td>
      <td>4.00</td>
      <td>4.2905</td>
      <td>80</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>437</th>
      <td>60</td>
      <td>2</td>
      <td>28.2</td>
      <td>112.00</td>
      <td>185</td>
      <td>113.8</td>
      <td>42.0</td>
      <td>4.00</td>
      <td>4.9836</td>
      <td>93</td>
    </tr>
    <tr>
      <th>438</th>
      <td>47</td>
      <td>2</td>
      <td>24.9</td>
      <td>75.00</td>
      <td>225</td>
      <td>166.0</td>
      <td>42.0</td>
      <td>5.00</td>
      <td>4.4427</td>
      <td>102</td>
    </tr>
    <tr>
      <th>439</th>
      <td>60</td>
      <td>2</td>
      <td>24.9</td>
      <td>99.67</td>
      <td>162</td>
      <td>106.6</td>
      <td>43.0</td>
      <td>3.77</td>
      <td>4.1271</td>
      <td>95</td>
    </tr>
    <tr>
      <th>440</th>
      <td>36</td>
      <td>1</td>
      <td>30.0</td>
      <td>95.00</td>
      <td>201</td>
      <td>125.2</td>
      <td>42.0</td>
      <td>4.79</td>
      <td>5.1299</td>
      <td>85</td>
    </tr>
    <tr>
      <th>441</th>
      <td>36</td>
      <td>1</td>
      <td>19.6</td>
      <td>71.00</td>
      <td>250</td>
      <td>133.2</td>
      <td>97.0</td>
      <td>3.00</td>
      <td>4.5951</td>
      <td>92</td>
    </tr>
  </tbody>
</table>
<p>442 rows × 10 columns</p>
</div>




```python
# Devide data into Train and Test Datasets
X_train, X_test, y_train, y_test = train_test_split(x_df,
                                                    y_df,
                                                    test_size=0.2,
                                                    random_state=66)

print(X_train)
```

         AGE  SEX   BMI     BP   S1     S2    S3    S4      S5   S6
    440   36    1  30.0   95.0  201  125.2  42.0  4.79  5.1299   85
    389   47    2  26.5   70.0  181  104.8  63.0  3.00  4.1897   70
    5     23    1  22.6   89.0  139   64.8  61.0  2.00  4.1897   68
    289   28    2  31.5   83.0  228  149.4  38.0  6.00  5.3132   83
    101   53    2  22.2  113.0  197  115.2  67.0  3.00  4.3041  100
    ..   ...  ...   ...    ...  ...    ...   ...   ...     ...  ...
    122   62    2  33.9  101.0  221  156.4  35.0  6.00  4.9972  103
    51    65    2  27.9  103.0  159   96.8  42.0  4.00  4.6151   86
    119   53    1  22.0   94.0  175   88.0  59.0  3.00  4.9416   98
    316   53    2  27.7   95.0  190  101.8  41.0  5.00  5.4638  101
    20    35    1  21.1   82.0  156   87.8  50.0  3.00  4.5109   95
    
    [353 rows x 10 columns]
    

# 3. Train Dataset


```python
# Import modules

from sklearn.linear_model import Ridge          # Algorithm
from sklearn.metrics import mean_squared_error  # Test score
from sklearn.externals import joblib            # Save trained models as *.pkl
import math
```

    /anaconda/envs/azureml_py38/lib/python3.8/site-packages/sklearn/externals/joblib/__init__.py:15: FutureWarning: sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. Please import this functionality directly from joblib, which can be installed with: pip install joblib. If this warning is raised when loading pickled models, you may need to re-serialize those models with scikit-learn 0.21+.
      warnings.warn(msg, category=FutureWarning)
    


```python
# Find the best parameter

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ]

for alpha in alphas:

    # Logs - Start value
    run = experiment.start_logging()
    run.log('alpha_value', alpha)

    # Train modles
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Mean Squared Error
    #mse = mean_squared_error(y_test, y_pred)
    
    # Squared Mean Squared Error
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    # Logs - Result
    run.log('rmse', rmse)

    # Print the result
    print('model_alpha={0}, rmse={1}'.format(alpha, rmse))

    # Set a name of model and path
    model_name = 'model_alpha_' + str(alpha) + '.pkl'  # Set a model name
    filename = 'outputs/' + model_name                 # Set a path
    joblib.dump(value=model, filename=filename)        # Export the model as a file

    # Upload the model file to Azure ML Service
    run.upload_file(name=model_name, path_or_stream=filename)


    #Logs - Complete
    run.complete()

    # Message for end
    print(f'{alpha} experiment completed!')

```

    model_alpha=0.1, rmse=56.605203313391435
    0.1 experiment completed!
    model_alpha=0.2, rmse=56.61060264545031
    0.2 experiment completed!
    model_alpha=0.3, rmse=56.61624324548362
    0.3 experiment completed!
    model_alpha=0.4, rmse=56.62210708871013
    0.4 experiment completed!
    model_alpha=0.5, rmse=56.628177342751385
    0.5 experiment completed!
    model_alpha=0.6, rmse=56.63443828302744
    0.6 experiment completed!
    model_alpha=0.7, rmse=56.64087521475942
    0.7 experiment completed!
    model_alpha=0.8, rmse=56.64747440101076
    0.8 experiment completed!
    model_alpha=0.9, rmse=56.65422299625313
    0.9 experiment completed!
    model_alpha=1.0, rmse=56.661108984990555
    1.0 experiment completed!
    


```python
experiment
```




<table style="width:100%"><tr><th>Name</th><th>Workspace</th><th>Report Page</th><th>Docs Page</th></tr><tr><td>diabetes-experiment</td><td>labmeow94ml</td><td><a href="https://ml.azure.com/experiments/id/1d421e1a-1f45-4122-b3e4-546840a29ee5?wsid=/subscriptions/27db5ec6-d206-4028-b5e1-6004dca5eeef/resourcegroups/rg94/workspaces/labmeow94ml&amp;tid=5fb256f0-fbf2-40d2-81d5-bac1b32c419d" target="_blank" rel="noopener">Link to Azure Machine Learning studio</a></td><td><a href="https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.experiment.Experiment?view=azure-ml-py" target="_blank" rel="noopener">Link to Documentation</a></td></tr></table>



# 4. Find the Best Model

## 4.1. Find the Best Model


```python
# Define the minimum RSME
minimum_rmse = None
minimum_rmse_runid = None
```


```python
# Compare models
for exp in experiment.get_runs():   # Get results of experiments
    run_metrics = exp.get_metrics();
    run_details = exp.get_details();

    run_rmse = run_metrics['rmse']
    run_id = run_details['runId']
```


```python
# Find and Download the Best Model
minimum_rmse = None
minimum_rmse_runid = None

for exp in experiment.get_runs():


    # Find the smallest value of RMSE
    if minimum_rmse is None:
        minimum_rmse = run_rmse
        minimum_rmse_runid = run_id
    else:
        if run_rmse <  minimum_rmse:
            minimum_rmse = run_rmse
            minimum_rmse_runid = run_id

print('Best run_id:  ' + minimum_rmse_runid)
print('Best run_id rmse:  ' + str(minimum_rmse))
```

    Best run_id:  bb1985de-73d7-46bc-b4bf-0d2bc0fb7a57
    Best run_id rmse:  56.64087521475942
    

## 4.2. Download the Best Model


```python
# Import module
from azureml.core import Run 
```


```python
# Download the model
best_run = Run(experiment=experiment, run_id=minimum_rmse_runid)
print(best_run.get_file_names())

best_run.download_file(name=str(best_run.get_file_names()[0]))
```

    ['model_alpha_0.7.pkl', 'outputs/.amlignore', 'outputs/.amlignore.amltmp', 'outputs/Model_alpha_0.1.pkl', 'outputs/Model_alpha_0.2.pkl', 'outputs/Model_alpha_0.3.pkl', 'outputs/Model_alpha_0.4.pkl', 'outputs/Model_alpha_0.5.pkl', 'outputs/Model_alpha_0.6.pkl', 'outputs/Model_alpha_0.7.pkl', 'outputs/Model_alpha_0.8.pkl', 'outputs/Model_alpha_0.9.pkl', 'outputs/Model_alpha_1.0.pkl']
    
