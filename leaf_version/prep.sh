#!/bin/bash

# file id of data stored on google drive
FILE_ID="1TVpUvGNGLgdZBDDSGQoOIzHz4U8TgHcT"

git checkout femnist_dataset
cd leaf_version

# Install gdown (for downloading data)
pip install gdown

# Download data
gdown --id $FILE_ID
mv -n femnist.zip ../data
unzip -f ../data/femnist.zip
rm ../data/femnist.zip

# Install Python dependencies
pip install -r ../requirements.txt