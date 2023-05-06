# 1. Preparation

## 1.1. Install Queue Package


```python
# Install the Package
!pip install azure-storage-queue==2.1.0
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: azure-storage-queue==2.1.0 in /usr/local/lib/python3.9/dist-packages (2.1.0)
    Requirement already satisfied: azure-common>=1.1.5 in /usr/local/lib/python3.9/dist-packages (from azure-storage-queue==2.1.0) (1.1.28)
    Requirement already satisfied: azure-storage-common~=2.1 in /usr/local/lib/python3.9/dist-packages (from azure-storage-queue==2.1.0) (2.1.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.9/dist-packages (from azure-storage-common~=2.1->azure-storage-queue==2.1.0) (2.27.1)
    Requirement already satisfied: cryptography in /usr/local/lib/python3.9/dist-packages (from azure-storage-common~=2.1->azure-storage-queue==2.1.0) (40.0.2)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.9/dist-packages (from azure-storage-common~=2.1->azure-storage-queue==2.1.0) (2.8.2)
    Requirement already satisfied: cffi>=1.12 in /usr/local/lib/python3.9/dist-packages (from cryptography->azure-storage-common~=2.1->azure-storage-queue==2.1.0) (1.15.1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.9/dist-packages (from python-dateutil->azure-storage-common~=2.1->azure-storage-queue==2.1.0) (1.16.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/dist-packages (from requests->azure-storage-common~=2.1->azure-storage-queue==2.1.0) (3.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/dist-packages (from requests->azure-storage-common~=2.1->azure-storage-queue==2.1.0) (2022.12.7)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/dist-packages (from requests->azure-storage-common~=2.1->azure-storage-queue==2.1.0) (1.26.15)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.9/dist-packages (from requests->azure-storage-common~=2.1->azure-storage-queue==2.1.0) (2.0.12)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.9/dist-packages (from cffi>=1.12->cryptography->azure-storage-common~=2.1->azure-storage-queue==2.1.0) (2.21)
    


```python
# Import modules
from azure.storage.queue import QueueService, QueueMessageFormat
```

## 1.2. Connect to Azure Queue


```python
# Define a Connection String of Azure Storage File
connect_str = 'DefaultEndpointsProtocol=https;AccountName=meow94storage;AccountKey=fA9o3JiS3RV09nYlLK+y+9mH4sP/uDgyOhTCKsXpoj2LMjCwvOLC67vY8PmwiwvgLO8GvhsZ/iWj+AStUq6zzQ==;EndpointSuffix=core.windows.net'

```


```python
# Create a Queue service and connect to Azure Queue
queue_name = 'queue-myqueue'
queue_service = QueueService(connection_string=connect_str)
```


```python
# Get the Queue
queue_service.create_queue(queue_name)
```




    False



# 2. Azure Queue

## 2.1. Define Encode and Decode


```python
# Defind an Encode
queue_service.encode_function = QueueMessageFormat.binary_base64encode

# Defind an Decode
queue_service.deconde_function = QueueMessageFormat.binary_base64decode
```


```python
# Import module
import base64
```

## 2.2. Encode 


```python
# Creat a message for test
message = 'Hello Python!'
print('Adding message: ' + message)
```

    Adding message: Hello Python!
    


```python
# Encode the test
message = base64.b64encode(message.encode('utf-8'))
```


```python
# Save the message in a queue
queue_service.put_message(queue_name, message)
```




    <azure.storage.queue.models.QueueMessage at 0x7f42b6b25e50>



## 2.3. Decode


```python
# Get the message by Decoding (The message is remaining in the queue)
messages = queue_service.peek_messages(queue_name)

for peeked_message in messages:
  message = base64.b64decode(peeked_message.content)
  print('Peeked message: ' + message.decode('utf-8'))
```

    Peeked message: SGVsbG8gUHl0aG9uIQ==
    


```python
# Decode and get the message (The message is removed in the queue)
messages = queue_service.get_messages(queue_name)

for msg in messages:
  message = base64.b64decode(msg.content)
  print('Got message: ' + message.decode('utf-8'))
```

    Got message: SGVsbG8gUHl0aG9uIQ==
    

# 3. Delete Queues


```python
# Delete Queues
queue_service.delete_queue(queue_name)
```




    True


