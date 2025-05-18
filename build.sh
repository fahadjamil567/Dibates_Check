#!/bin/bash

# Ensure python3.9 is available and create symlink if needed
if command -v python3.9 &> /dev/null; then
    echo "Python 3.9 found"
else
    echo "Python 3.9 not found, creating symlink"
    ln -s $(which python3) /usr/local/bin/python3.9
fi

# Ensure pip is available
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.9 get-pip.py --force-reinstall

# Install dependencies
python3.9 -m pip install --upgrade pip
python3.9 -m pip install -r requirements-vercel.txt

# Create necessary directories
mkdir -p .vercel/cache 