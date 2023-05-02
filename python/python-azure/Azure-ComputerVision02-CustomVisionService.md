# 1. Preparation


```python
# Azure custom vision download and install

!pip install azure-cognitiveservices-vision-customvision
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: azure-cognitiveservices-vision-customvision in /usr/local/lib/python3.9/dist-packages (3.1.0)
    Requirement already satisfied: msrest>=0.5.0 in /usr/local/lib/python3.9/dist-packages (from azure-cognitiveservices-vision-customvision) (0.7.1)
    Requirement already satisfied: azure-common~=1.1 in /usr/local/lib/python3.9/dist-packages (from azure-cognitiveservices-vision-customvision) (1.1.28)
    Requirement already satisfied: isodate>=0.6.0 in /usr/local/lib/python3.9/dist-packages (from msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (0.6.1)
    Requirement already satisfied: requests~=2.16 in /usr/local/lib/python3.9/dist-packages (from msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (2.27.1)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/dist-packages (from msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (2022.12.7)
    Requirement already satisfied: requests-oauthlib>=0.5.0 in /usr/local/lib/python3.9/dist-packages (from msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (1.3.1)
    Requirement already satisfied: azure-core>=1.24.0 in /usr/local/lib/python3.9/dist-packages (from msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (1.26.4)
    Requirement already satisfied: six>=1.11.0 in /usr/local/lib/python3.9/dist-packages (from azure-core>=1.24.0->msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (1.16.0)
    Requirement already satisfied: typing-extensions>=4.3.0 in /usr/local/lib/python3.9/dist-packages (from azure-core>=1.24.0->msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (4.5.0)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.9/dist-packages (from requests~=2.16->msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (2.0.12)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/dist-packages (from requests~=2.16->msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (1.26.15)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/dist-packages (from requests~=2.16->msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (3.4)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.9/dist-packages (from requests-oauthlib>=0.5.0->msrest>=0.5.0->azure-cognitiveservices-vision-customvision) (3.2.2)
    


```python
# Import modules

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
```


```python
# Connect to Microsoft Azure Custom Vision Service

endpoint = 'https://meow94customvision.cognitiveservices.azure.com/'
training_key = 'f9f9321d3a5f4284abab538f5b4b6fb9'
```


```python
publish_iteration_name = 'GreatMeowModel'

# Make credentials using Training key
credentials = ApiKeyCredentials(in_headers={'Training-key': training_key}) 
```


```python
# Load the Custom Vision service
trainer = CustomVisionTrainingClient(endpoint, credentials)
```

# 2. Cognitive Custom Vision


```python
# Create a new project
print('Creating a project....')
project_name = 'GreatMeow94'
project = trainer.create_project(project_name)
```

    Creating a project....
    


```python
# Create tags

blacknoodle_tag = trainer.create_tag(project.id, 'blacknoodle')  # create_tag(Project ID, Tag names)
rednoodle_tag = trainer.create_tag(project.id, 'rednoodle')
sweetandsourpork_tag = trainer.create_tag(project.id, 'sweetandsourpork')
```


```python
# Use 'Time' module to repeat 'while Statement' every 10 seconds
import time

# Start training
print ('Training......')
iteration = trainer.train_project(project.id)
while (iteration.status != 'Completed'):
  iteration = trainer.get_iteration(project.id, iteration.id)
  print('Training status: ' + iteration.status) 
  print('Waiting for 30 seconds.....')
  time.sleep(30)   # Run a 'while' statement every 30 seconds
```

    Training......
    Training status: Training
    Waiting for 30 seconds.....
    Training status: Training
    Waiting for 30 seconds.....
    Training status: Training
    Waiting for 30 seconds.....
    Training status: Training
    Waiting for 30 seconds.....
    Training status: Training
    Waiting for 30 seconds.....
    Training status: Training
    Waiting for 30 seconds.....
    Training status: Training
    Waiting for 30 seconds.....
    Training status: Training
    Waiting for 30 seconds.....
    Training status: Training
    Waiting for 30 seconds.....
    Training status: Completed
    Waiting for 30 seconds.....
    


```python
# Import module
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
```


```python
prediction_key = '1ae879f101af4b4e906a068f1da15417'
prediction_endpoint = 'https://meow94customvision-prediction.cognitiveservices.azure.com/'


prediction_credential = ApiKeyCredentials(
    in_headers={'Prediction-key': prediction_key})
```


```python
predictor = CustomVisionPredictionClient(prediction_endpoint,
                                         prediction_credential)
```


```python
target_image_url = 'https://www.newiki.net/w/images/thumb/d/d9/Jjajangmyeon.jpg/1200px-Jjajangmyeon.jpg'
result = predictor.classify_image_url(project.id,
                                      'Iteration1',
                                      target_image_url)
```


```python
for prediction in result.predictions:
  print('\t' + prediction.tag_name + 
        ': {0:.4f}%'.format(prediction.probability * 100))
```
