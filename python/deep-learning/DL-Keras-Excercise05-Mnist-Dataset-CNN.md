# 0. Overview

- CNN (Convolutional Neural Network)
- Dataset: Mnist Dataset

# 1. Preparation


```python
# Import modules
from keras import models
from keras import layers
```

# 2. Create a Neural Network

- Layers
 - Max Pooling Layer: Reduce the number of layers by using the maximum numbers
 - Flatten Layer: Flatten the final layer into 1-D


## 2.1. Max Pooling Layer


```python
# Create a model

model = models.Sequential()

# Input layer - Convolutional Network
model.add(layers.Conv2D(32,      # Number of filters
                        (3, 3),  # Size of filter
                        activation='relu',
                        input_shape=(28, 28, 1)))  # (28, 28): The size of image,  (1): Black&White

# Hidden layer
model.add(layers.MaxPooling2D(2,2))  # Size of filter

# Repeat of Input and Hidden layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2,2))
```


```python
# Summary of model
model.summary()

# Result
#  - Parameters: the Total Parameters are more complex than the model with Densor layer
#  - Output Shape: size of Input is smaller and smaller
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                     
     max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 11, 11, 32)        9248      
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 5, 5, 32)         0         
     2D)                                                             
                                                                     
     conv2d_2 (Conv2D)           (None, 3, 3, 32)          9248      
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 1, 1, 32)         0         
     2D)                                                             
                                                                     
    =================================================================
    Total params: 18,816
    Trainable params: 18,816
    Non-trainable params: 0
    _________________________________________________________________
    

## 2.2. Flatten Layer


```python
# Hidden Layer - Flatten layer
model.add(layers.Flatten()) 

# Hidden Layer - Dense layer
model.add(layers.Dense(64, activation='relu'))

# Output Layer
model.add(layers.Dense(10, activation='softmax'))
```


```python
# Summary the model
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                     
     max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 11, 11, 32)        9248      
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 5, 5, 32)         0         
     2D)                                                             
                                                                     
     conv2d_2 (Conv2D)           (None, 3, 3, 32)          9248      
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 1, 1, 32)         0         
     2D)                                                             
                                                                     
     flatten (Flatten)           (None, 32)                0         
                                                                     
     dense (Dense)               (None, 64)                2112      
                                                                     
     dense_1 (Dense)             (None, 10)                650       
                                                                     
    =================================================================
    Total params: 21,578
    Trainable params: 21,578
    Non-trainable params: 0
    _________________________________________________________________
    

# 3. Get Data


```python
# Import modules
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
```


```python
# Get data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11490434/11490434 [==============================] - 1s 0us/step
    


```python
# Check shape 
train_images.shape
```




    (60000, 28, 28)




```python
# Check data
train_images[0]
```




    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,
             18,  18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170,
            253, 253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253,
            253, 253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253,
            253, 198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253,
            205,  11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,
             90,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253,
            190,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190,
            253,  70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
            241, 225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             81, 240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,  45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,  46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39,
            148, 229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221,
            253, 253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253,
            253, 253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253,
            195,  80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,
             11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0]], dtype=uint8)



# 4. Scaling


```python
# Increase the number of dimensions to match the shape to the Input Layer(Convolutional Layer)
train_data = train_images.reshape((60000,  # Number of data 
                                    28,     # X-axis
                                    28,     # Y-axis
                                    1))     # Dimension

train_data = train_data.astype('float32') / 255   # Scaling
 

test_data = test_images.reshape((10000,28,28,1))
test_data = test_data.astype('float32') / 255

train_target = to_categorical(train_labels)
test_target = to_categorical(test_labels)
```


```python
# Check a shape
train_data.shape
```




    (60000, 28, 28, 1)




```python
# Check data
train_data[0]
```




    array([[[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.01176471],
            [0.07058824],
            [0.07058824],
            [0.07058824],
            [0.49411765],
            [0.53333336],
            [0.6862745 ],
            [0.10196079],
            [0.6509804 ],
            [1.        ],
            [0.96862745],
            [0.49803922],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.11764706],
            [0.14117648],
            [0.36862746],
            [0.6039216 ],
            [0.6666667 ],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.88235295],
            [0.6745098 ],
            [0.99215686],
            [0.9490196 ],
            [0.7647059 ],
            [0.2509804 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.19215687],
            [0.93333334],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.9843137 ],
            [0.3647059 ],
            [0.32156864],
            [0.32156864],
            [0.21960784],
            [0.15294118],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.07058824],
            [0.85882354],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.7764706 ],
            [0.7137255 ],
            [0.96862745],
            [0.94509804],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.3137255 ],
            [0.6117647 ],
            [0.41960785],
            [0.99215686],
            [0.99215686],
            [0.8039216 ],
            [0.04313726],
            [0.        ],
            [0.16862746],
            [0.6039216 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.05490196],
            [0.00392157],
            [0.6039216 ],
            [0.99215686],
            [0.3529412 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.54509807],
            [0.99215686],
            [0.74509805],
            [0.00784314],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.04313726],
            [0.74509805],
            [0.99215686],
            [0.27450982],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.13725491],
            [0.94509804],
            [0.88235295],
            [0.627451  ],
            [0.42352942],
            [0.00392157],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.31764707],
            [0.9411765 ],
            [0.99215686],
            [0.99215686],
            [0.46666667],
            [0.09803922],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.1764706 ],
            [0.7294118 ],
            [0.99215686],
            [0.99215686],
            [0.5882353 ],
            [0.10588235],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.0627451 ],
            [0.3647059 ],
            [0.9882353 ],
            [0.99215686],
            [0.73333335],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.9764706 ],
            [0.99215686],
            [0.9764706 ],
            [0.2509804 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.18039216],
            [0.50980395],
            [0.7176471 ],
            [0.99215686],
            [0.99215686],
            [0.8117647 ],
            [0.00784314],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.15294118],
            [0.5803922 ],
            [0.8980392 ],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.98039216],
            [0.7137255 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.09411765],
            [0.44705883],
            [0.8666667 ],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.7882353 ],
            [0.30588236],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.09019608],
            [0.25882354],
            [0.8352941 ],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.7764706 ],
            [0.31764707],
            [0.00784314],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.07058824],
            [0.67058825],
            [0.85882354],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.7647059 ],
            [0.3137255 ],
            [0.03529412],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.21568628],
            [0.6745098 ],
            [0.8862745 ],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.95686275],
            [0.52156866],
            [0.04313726],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.53333336],
            [0.99215686],
            [0.99215686],
            [0.99215686],
            [0.83137256],
            [0.5294118 ],
            [0.5176471 ],
            [0.0627451 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]],
    
           [[0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]]], dtype=float32)



# 5. Fit Data


```python
# Compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```


```python
# Fit data
model.fit(train_data,
          train_target,
          epochs=5,
          batch_size=64)
```

    Epoch 1/5
    938/938 [==============================] - 15s 5ms/step - loss: 0.3534 - accuracy: 0.8890
    Epoch 2/5
    938/938 [==============================] - 4s 4ms/step - loss: 0.1141 - accuracy: 0.9650
    Epoch 3/5
    938/938 [==============================] - 4s 4ms/step - loss: 0.0817 - accuracy: 0.9751
    Epoch 4/5
    938/938 [==============================] - 5s 5ms/step - loss: 0.0644 - accuracy: 0.9800
    Epoch 5/5
    938/938 [==============================] - 5s 6ms/step - loss: 0.0544 - accuracy: 0.9836
    




    <keras.callbacks.History at 0x7fa2f02096c0>



# 6. Evaluation


```python
test_loss, test_acc = model.evaluate(test_data, test_target)
```

    313/313 [==============================] - 1s 3ms/step - loss: 0.0778 - accuracy: 0.9747
    


```python
test_acc
```




    0.9746999740600586




```python

```