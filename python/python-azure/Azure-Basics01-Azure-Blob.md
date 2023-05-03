# 1. Preparation



```python
# Install Azure Storage(Blob) and Identity

!pip install azure-storage-blob azure-identity
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: azure-storage-blob in /usr/local/lib/python3.9/dist-packages (12.16.0)
    Requirement already satisfied: azure-identity in /usr/local/lib/python3.9/dist-packages (1.12.0)
    Requirement already satisfied: isodate>=0.6.1 in /usr/local/lib/python3.9/dist-packages (from azure-storage-blob) (0.6.1)
    Requirement already satisfied: azure-core<2.0.0,>=1.26.0 in /usr/local/lib/python3.9/dist-packages (from azure-storage-blob) (1.26.4)
    Requirement already satisfied: cryptography>=2.1.4 in /usr/local/lib/python3.9/dist-packages (from azure-storage-blob) (40.0.2)
    Requirement already satisfied: typing-extensions>=4.0.1 in /usr/local/lib/python3.9/dist-packages (from azure-storage-blob) (4.5.0)
    Requirement already satisfied: msal-extensions<2.0.0,>=0.3.0 in /usr/local/lib/python3.9/dist-packages (from azure-identity) (1.0.0)
    Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.9/dist-packages (from azure-identity) (1.16.0)
    Requirement already satisfied: msal<2.0.0,>=1.12.0 in /usr/local/lib/python3.9/dist-packages (from azure-identity) (1.22.0)
    Requirement already satisfied: requests>=2.18.4 in /usr/local/lib/python3.9/dist-packages (from azure-core<2.0.0,>=1.26.0->azure-storage-blob) (2.27.1)
    Requirement already satisfied: cffi>=1.12 in /usr/local/lib/python3.9/dist-packages (from cryptography>=2.1.4->azure-storage-blob) (1.15.1)
    Requirement already satisfied: PyJWT[crypto]<3,>=1.0.0 in /usr/local/lib/python3.9/dist-packages (from msal<2.0.0,>=1.12.0->azure-identity) (2.6.0)
    Requirement already satisfied: portalocker<3,>=1.0 in /usr/local/lib/python3.9/dist-packages (from msal-extensions<2.0.0,>=0.3.0->azure-identity) (2.7.0)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.9/dist-packages (from cffi>=1.12->cryptography>=2.1.4->azure-storage-blob) (2.21)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/dist-packages (from requests>=2.18.4->azure-core<2.0.0,>=1.26.0->azure-storage-blob) (1.26.15)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.9/dist-packages (from requests>=2.18.4->azure-core<2.0.0,>=1.26.0->azure-storage-blob) (2.0.12)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/dist-packages (from requests>=2.18.4->azure-core<2.0.0,>=1.26.0->azure-storage-blob) (2022.12.7)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/dist-packages (from requests>=2.18.4->azure-core<2.0.0,>=1.26.0->azure-storage-blob) (3.4)
    


```python
# Import modules

import os, uuid   # os: Manage directories
from azure.identity import DefaultAzureCredential  # Authentification
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient   # Packages for Blob
```

# 2. Microsoft Azure Blob(Storage)

## 2.1. Connect to Blob Service


```python
# Use Connection String from Azure
connect_str = 'DefaultEndpointsProtocol=https;AccountName=meow94storage;AccountKey=fA9o3JiS3RV09nYlLK+y+9mH4sP/uDgyOhTCKsXpoj2LMjCwvOLC67vY8PmwiwvgLO8GvhsZ/iWj+AStUq6zzQ==;EndpointSuffix=core.windows.net'

# Connect to Blobk service
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
```

## 2.2. Create a Container using Blob Service


```python
# Name a container
container_name = 'test3container'

# Create a container
container_client = blob_service_client.create_container(container_name)
```

## 2.3. Create a Local Directory uploading to Blob Service


```python
# Create a path in the local storage
local_path = './data'
os.mkdir(local_path)    # mkdir = make a directory

```


```python
# Make a text file in the local directory
local_file_name = 'welcome' + '.txt'
upload_file_path = os.path.join(local_path, local_file_name)
```


```python
# Create/Open a file -> Edit file -> Close File
file = open(upload_file_path, mode='w')  # If the file is not existed, create a new file and open with 'w' mode
file.write('Welcome to Python!')
file.close()  # close 안하면 파일 잘 깨짐
```

## 2.4. Upload files to Blob


```python
# Get Blob service
blob_client = blob_service_client.get_blob_client(container=container_name,
                                                  blob=local_file_name)

# Upload files
with open(file=upload_file_path, mode='rb') as data:
  blob_client.upload_blob(data)

# File Modes
# w: text written mode
# r: text read mode
# wb: binary written mode
# rb: binary read mode
```


```python
print ('\nListing blobs')

blob_list = container_client.list_blobs()
for blob in blob_list:
  print('\t' + blob.name)  # \t: tab
```

    
    Listing blobs
    	welcome.txt
    


```python
blob
```




    {'name': 'welcome.txt', 'container': 'test3container', 'snapshot': None, 'version_id': None, 'is_current_version': None, 'blob_type': <BlobType.BLOCKBLOB: 'BlockBlob'>, 'metadata': {}, 'encrypted_metadata': None, 'last_modified': datetime.datetime(2023, 4, 20, 1, 21, 50, tzinfo=datetime.timezone.utc), 'etag': '0x8DB413D9E5E17B8', 'size': 18, 'content_range': None, 'append_blob_committed_block_count': None, 'is_append_blob_sealed': None, 'page_blob_sequence_number': None, 'server_encrypted': True, 'copy': {'id': None, 'source': None, 'status': None, 'progress': None, 'completion_time': None, 'status_description': None, 'incremental_copy': None, 'destination_snapshot': None}, 'content_settings': {'content_type': 'application/octet-stream', 'content_encoding': None, 'content_language': None, 'content_md5': bytearray(b'\xca\x0f\xe68\xd1\xb1\xfa"\x8b\x99&\x82\xf9$o\xce'), 'content_disposition': None, 'cache_control': None}, 'lease': {'status': 'unlocked', 'state': 'available', 'duration': None}, 'blob_tier': 'Hot', 'rehydrate_priority': None, 'blob_tier_change_time': None, 'blob_tier_inferred': True, 'deleted': None, 'deleted_time': None, 'remaining_retention_days': None, 'creation_time': datetime.datetime(2023, 4, 20, 1, 21, 50, tzinfo=datetime.timezone.utc), 'archive_status': None, 'encryption_key_sha256': None, 'encryption_scope': None, 'request_server_encrypted': None, 'object_replication_source_properties': [], 'object_replication_destination_policy': None, 'last_accessed_on': None, 'tag_count': None, 'tags': None, 'immutability_policy': {'expiry_time': None, 'policy_mode': None}, 'has_legal_hold': None, 'has_versions_only': None}



## 2.5. Download files to Colab Local from Blob Service


```python
# Set a download path
download_file_path = os.path.join(local_path, 
                                  str.replace(local_file_name, '.txt', '_downloaded.txt'))

print(download_file_path)

container_client = blob_service_client.get_container_client(container=container_name)
```

    ./data/welcome_downloaded.txt
    


```python
# Open the file after download from Blob to Colab Local
with open(file=download_file_path, mode='wb') as download_file:
  container_client.download_blob(blob.name).readall()
```

## 2.6. Delete Job Histories


```python
print('\nPress the "y" to begin cleaning up')
if input() == 'y':

  # Delete Azure Containers
  print('Deleting blob containers....')
  container_client.delete_container()

  # Delete Colab local directories and downloaded files
  print('Deleting the local sources and downloaded files.....')
  os.remove(upload_file_path)
  os.remove(download_file_path)
  os.rmdir(local_path)

  # Pring when it's done
  print('Done!')
```

    
    Press the "y" to begin cleaning up
    y
    Deleting blob containers....
    Deleting the local sources and downloaded files.....
    Done
    
