#!/bin/bash

# file id of data stored on google drive
FILE_ID="1TVpUvGNGLgdZBDDSGQoOIzHz4U8TgHcT"

git checkout femnist_dataset

# Install gdown (for downloading data)
pip install gdown

# Download data
cd data
gdown --id $FILE_ID
unzip -o femnist.zip
rm femnist.zip
cd ..

# Install Python dependencies
pip install -r requirements.txt