#!/bin/bash

# Download and install Python 3.9
curl -O https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz
tar -xf Python-3.9.18.tgz
cd Python-3.9.18
./configure --enable-optimizations
make -j $(nproc)
sudo make altinstall

# Create and activate virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements-vercel.txt 