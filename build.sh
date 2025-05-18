#!/bin/bash
python3 -m pip install --upgrade pip
pip install --no-cache-dir --no-deps -r requirements-vercel.txt
pip install --no-cache-dir -r requirements-vercel.txt 