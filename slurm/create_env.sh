#!/bin/bash

module load python/3.10
module load scipy-stack/2023a

cd ..
virtualenv --no-download env
source env/bin/activate
pip install -r requirements.txt
