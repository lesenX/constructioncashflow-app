#!/bin/bash

# 安装系统依赖
apt-get update
apt-get install -y libgl1-mesa-glx

# 升级pip并安装Python包
pip install --upgrade pip
pip install -r requirements.txt
