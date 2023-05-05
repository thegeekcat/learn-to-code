# 1. Preparation


```python
# Install 'Azure Storage File'
!pip install azure-storage-file
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting azure-storage-file
      Downloading azure_storage_file-2.1.0-py2.py3-none-any.whl (36 kB)
    Collecting azure-common>=1.1.5
      Downloading azure_common-1.1.28-py2.py3-none-any.whl (14 kB)
    Collecting azure-storage-common~=2.1
      Downloading azure_storage_common-2.1.0-py2.py3-none-any.whl (47 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m47.8/47.8 kB[0m [31m2.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: python-dateutil in /usr/local/lib/python3.9/dist-packages (from azure-storage-common~=2.1->azure-storage-file) (2.8.2)
    Requirement already satisfied: cryptography in /usr/local/lib/python3.9/dist-packages (from azure-storage-common~=2.1->azure-storage-file) (40.0.2)
    Requirement already satisfied: requests in /usr/local/lib/python3.9/dist-packages (from azure-storage-common~=2.1->azure-storage-file) (2.27.1)
    Requirement already satisfied: cffi>=1.12 in /usr/local/lib/python3.9/dist-packages (from cryptography->azure-storage-common~=2.1->azure-storage-file) (1.15.1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.9/dist-packages (from python-dateutil->azure-storage-common~=2.1->azure-storage-file) (1.16.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/dist-packages (from requests->azure-storage-common~=2.1->azure-storage-file) (2022.12.7)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/dist-packages (from requests->azure-storage-common~=2.1->azure-storage-file) (1.26.15)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/dist-packages (from requests->azure-storage-common~=2.1->azure-storage-file) (3.4)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.9/dist-packages (from requests->azure-storage-common~=2.1->azure-storage-file) (2.0.12)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.9/dist-packages (from cffi>=1.12->cryptography->azure-storage-common~=2.1->azure-storage-file) (2.21)
    Installing collected packages: azure-common, azure-storage-common, azure-storage-file
    Successfully installed azure-common-1.1.28 azure-storage-common-2.1.0 azure-storage-file-2.1.0
    


```python
# Import modules
from azure.storage.file import FileService 
```

# 2. Microsoft File Storage

## 2.1. Mount a new Azure File Storage


```python
# Mount a new Azure File Storage as a drive
file_service = FileService(account_name='meow94storage',
                           account_key='fA9o3JiS3RV09nYlLK+y+9mH4sP/uDgyOhTCKsXpoj2LMjCwvOLC67vY8PmwiwvgLO8GvhsZ/iWj+AStUq6zzQ==')

```


```python
#Add a new Fileshare
file_service.create_share('myshare')
```




    True



## 2.2. Add a new Directory to Azure File Share


```python
# Azure file Share에 디렉토리 생성
file_service.create_directory('myshare', 'directory_01')  # create_directory('name of file share', 'Name of new directory')
```




    True



## 2.3. Upload to Azure File Storage


```python
# Import module related to content settings
from azure.storage.file import ContentSettings
```


```python
file_service.create_file_from_path(
    'myshare',       # Name of File Share to upload
    'directory_01',  # Folder path to upload
    'meow1.png',     # Set a name of file to upload
    '1.png'          # Name of file to upload on local storage
)
```

## 2.3. Check a Directory


```python
# Get a list of files in Directory
generator = file_service.list_directories_and_files('myshare')  # list_directories_and_files(Name of FileShare)

# Print contents of files
for file_or_dir in generator:
  print(file_or_dir.name)

```

    meow1.png
    myfile
    directory_01
    

## 2.4. Download files from Azure File Storage to Local storage


```python
# Download
file_service.get_file_to_path('myshare',            # name of File Share
                              None,                 # No specific paths
                              'meow1.png',          # Target file name to download on Azure server
                              'meow1_download.png') # File name to save on local storage
```




    <azure.storage.file.models.File at 0x7f117c97c8e0>



## 2.5. Snapshot 


```python
# Snapshot: Snapshot is a backup image to restore files and directories
```


```python
# Create a metadata for test
metadata = {'foo' : 'bar'}
```


```python
# Create a Snapshot
snapshot = file_service.snapshot_share('myshare',          # Set a storage to make a snapshot
                                       metadata=metadata)  # Metadata for snapshot
```


```python
# Get a list of Snapshots from Azure
share = list(file_service.list_shares(include_snapshots=True))
```


```python
# Get a list of directories and files

directories_and_files = list(file_service.list_directories_and_files('myshare',   # Name of File Share
                                                                     snapshot='2023-04-20T05:49:37.0000000Z')) # Name of Snapshot
# Print a list of Files and Directories
for file_or_dir in directories_and_files:
  print(file_or_dir.name)
```

    meow1.png
    myfile
    directory_01
    

## 2.6. Delete


```python
# Delete Snapshots
file_service.delete_share('myshare',
                          snapshot='2023-04-20T05:49:37.0000000Z')
```




    True




```python
# Delete File Shares
file_service.delete_file('myshare',
                         None,     
                         '1.png')  # File name to be deleted
```


```python
# Delete a directory in a Snapshot image
file_service.delete_directory('myshare', # Name of File Share
                              'directory_01')  # Folders to be deleted
```


```python
# Delete File Share
file_service.delete_directory('myshare')  # Name of File Share to be deleted
```
