# 1. Preparation


```python
# Import modules
from keras.datasets import boston_housing
```


```python
# Defind dataset
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
```


```python
# Check a shape
print('Train Data: ', train_data.shape)
print('Test  Data: ', test_data.shape)

# Result
# - Number of data: 404
# - Number of Categories: 13
```

    Train Data:  (404, 13)
    Test  Data:  (102, 13)
    


```python
# Check Label data
print(train_labels[:10])
```

    [15.2 42.3 50.  21.1 17.7 18.5 11.3 15.6 15.6 14.4]
    


```python
print(train_data[0])

# Result: A range of numbers are vary -> Need scaling
```

    [  1.23247   0.        8.14      0.        0.538     6.142    91.7
       3.9769    4.      307.       21.      396.9      18.72   ]
    

# 2. Pre-Processing

## 2.1. Standard Scaling by Manual


```python
# Get Mean by columns

#mean = train_data.mean()  # If there's no axis, a result shows a mean value of all data
mean = train_data.mean(axis=0)  # Get mean by columns

print('Mean: ', mean)

```

    Mean:  [3.74511057e+00 1.14801980e+01 1.11044307e+01 6.18811881e-02
     5.57355941e-01 6.26708168e+00 6.90106436e+01 3.74027079e+00
     9.44059406e+00 4.05898515e+02 1.84759901e+01 3.54783168e+02
     1.27408168e+01]
    


```python
# std by columns
train_data -= mean  # train_data = train_data - mean

std = train_data.std(axis=0)
std
```




    array([9.22929073e+00, 2.37382770e+01, 6.80287253e+00, 2.40939633e-01,
           1.17147847e-01, 7.08908627e-01, 2.79060634e+01, 2.02770050e+00,
           8.68758849e+00, 1.66168506e+02, 2.19765689e+00, 9.39946015e+01,
           7.24556085e+00])




```python
train_data /= std
train_data[0]
```




    array([-0.27224633, -0.48361547, -0.43576161, -0.25683275, -0.1652266 ,
           -0.1764426 ,  0.81306188,  0.1166983 , -0.62624905, -0.59517003,
            1.14850044,  0.44807713,  0.8252202 ])



# 3. Create a Neural Network


```python
# Import modules
from keras import models
from keras import layers
```


```python
# Create a model

model = models.Sequential()

# Defind a function to create a model 
# -> Reason: To use K-Fold, initializing models for each epoch is needed, so make a function to initial models easily

def build_model():
  # Input layer
  model.add(layers.Dense(64,
                        activation='relu',
                        input_shape=(train_data.shape[1],)))   # Use train_data's shape instead of using (10000,) 

  # Hidden layer
  model.add(layers.Dense(64,
                        activation='relu'))

  # Output layer
  model.add(layers.Dense(1))  # No activation value as it's a linear regression model

  # Compile
  model.compile(optimizer='rmsprop',
                loss='mse',
                metrics=['mse'])
  
  return model
```

# 4. Validate the Model using K-Fold

- K-Fold = 4
  - V: Validation Data, T: Train Data
  - Fold 1: [ V ][ T ][ T ][ T ]
  - Fold 2: [ T ][ V ][ T ][ T ]
  - Fold 3: [ T ][ T ][ V ][ T ]
  - Fold 4: [ T ][ T ][ T ][ V ]
  


```python
# Import modules
import numpy as np
```


```python
# Number of K-Fold
k = 4

# Number of data in each Fold
num_val_samples = len(train_data) // 4  # 
print(num_val_samples)
```

    101
    


```python
# Make an empty list for results
all_scores = []


for i in range(k):  # Repeat 4 times
  print('The curruent Fold in progress is: # ', i)

  # Find the start and end values in each Fold
  # val_data = i * num_val_samples            # Start
  # val_data_end = (i + 1) * num_val_samples  # End

  # Prepare Validation Data
  val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
  val_labels = train_labels[i * num_val_samples: (i + 1) * num_val_samples]

  # Prepare Train Data
  # Train Data can be 1 or 2
  #   - Fold 1: [ V ][ T ][ T ][ T ]  -> One group:   ( v )[ T ][ T ][ T ]
  #   - Fold 2: [ T ][ V ][ T ][ T ]  -> Two groups:  [ T ]( v )[ T ][ T ]
  #   - Fold 3: [ T ][ T ][ V ][ T ]  -> Two groups:  [ T ][ T ]( v )[ T ]
  #   - Fold 4: [ T ][ T ][ T ][ V ]  -> One group:   [ T ][ T ][ T ]( v )
  data1 = train_data[:i * num_val_samples]
  #print(' - Train group 1: ', 0, '-', i * num_val_samples)
  data2 = train_data[(i+1) * num_val_samples :]
  #print(' - Train group 2: ', (i+1) * num_val_samples, '-', 404)

  # Prepare Label Data
  data1_labels = train_labels[:i * num_val_samples]
  data2_labels = train_labels[(i+1) * num_val_samples :]

  # Get train data and labels
  partial_train_data = np.concatenate([data1, data2], axis=0) # Combine data1 and data2 based by columns
  partial_train_labels = np.concatenate([data1_labels, data2_labels], axis=0)

  # Create a neural network
  model = build_model()
  model.summary()   # Summary the model
  model.fit(partial_train_data, 
            partial_train_labels, 
            epochs=100, 
            batch_size=128,
            verbose=0)  # Option to print out training processes as training times are over 400 times
  
  # Validate the model
  val_mse, val_mae = model.evaluate(val_data, val_labels)
  print(val_mse, val_mae)
  all_scores.append(val_mae)

```

    The curruent Fold in progress is: #  0
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 64)                896       
                                                                     
     dense_1 (Dense)             (None, 64)                4160      
                                                                     
     dense_2 (Dense)             (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 5,121
    Trainable params: 5,121
    Non-trainable params: 0
    _________________________________________________________________
    4/4 [==============================] - 0s 7ms/step - loss: 8.6604 - mse: 8.6604
    8.660383224487305 8.660383224487305
    The curruent Fold in progress is: #  1
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 64)                896       
                                                                     
     dense_1 (Dense)             (None, 64)                4160      
                                                                     
     dense_2 (Dense)             (None, 1)                 65        
                                                                     
     dense_3 (Dense)             (None, 64)                128       
                                                                     
     dense_4 (Dense)             (None, 64)                4160      
                                                                     
     dense_5 (Dense)             (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 9,474
    Trainable params: 9,474
    Non-trainable params: 0
    _________________________________________________________________
    4/4 [==============================] - 0s 6ms/step - loss: 10.2357 - mse: 10.2357
    10.235672950744629 10.235672950744629
    The curruent Fold in progress is: #  2
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 64)                896       
                                                                     
     dense_1 (Dense)             (None, 64)                4160      
                                                                     
     dense_2 (Dense)             (None, 1)                 65        
                                                                     
     dense_3 (Dense)             (None, 64)                128       
                                                                     
     dense_4 (Dense)             (None, 64)                4160      
                                                                     
     dense_5 (Dense)             (None, 1)                 65        
                                                                     
     dense_6 (Dense)             (None, 64)                128       
                                                                     
     dense_7 (Dense)             (None, 64)                4160      
                                                                     
     dense_8 (Dense)             (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 13,827
    Trainable params: 13,827
    Non-trainable params: 0
    _________________________________________________________________
    4/4 [==============================] - 0s 5ms/step - loss: 12.4885 - mse: 12.4885
    12.488473892211914 12.488473892211914
    The curruent Fold in progress is: #  3
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 64)                896       
                                                                     
     dense_1 (Dense)             (None, 64)                4160      
                                                                     
     dense_2 (Dense)             (None, 1)                 65        
                                                                     
     dense_3 (Dense)             (None, 64)                128       
                                                                     
     dense_4 (Dense)             (None, 64)                4160      
                                                                     
     dense_5 (Dense)             (None, 1)                 65        
                                                                     
     dense_6 (Dense)             (None, 64)                128       
                                                                     
     dense_7 (Dense)             (None, 64)                4160      
                                                                     
     dense_8 (Dense)             (None, 1)                 65        
                                                                     
     dense_9 (Dense)             (None, 64)                128       
                                                                     
     dense_10 (Dense)            (None, 64)                4160      
                                                                     
     dense_11 (Dense)            (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 18,180
    Trainable params: 18,180
    Non-trainable params: 0
    _________________________________________________________________
    4/4 [==============================] - 0s 4ms/step - loss: 7.8128 - mse: 7.8128
    7.812751770019531 7.812751770019531
    


```python
# Check scores
all_scores
```




    [8.660383224487305, 10.235672950744629, 12.488473892211914, 7.812751770019531]




```python
# Need visualization by myself
```


```python

```
