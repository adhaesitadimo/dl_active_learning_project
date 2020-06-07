#!/bin/bash

mkdir packages
git clone https://github.com/IINemo/libact.git@seq ./packages/libact
git clone https://github.com/IINemo/active_learning_toolbox.git@seq ./packages/active_learning_toolbox

pip install -e packages/libact
pip install -e packages/active_learning_toolbox
pip install -r requirements.txt