# Drawing Robot With Large Language Model

This repository is the implementation of the class: Robot learning 101 (JBNU 2025 fall, M.S course)

## Setup

Step1: Prepare dataset (I used QuickDraw dataset)

Step2: `python dataset_generate.py`

if you want to check the dataset, run `python check_dataset.py`

## Training

`python train.py --db_root "your db_root" --dbname "e.g. sunny" --use_reproj --smoothness --reproj_option OP`
