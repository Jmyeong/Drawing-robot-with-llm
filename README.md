# Drawing Robot With Large Language Model

This repository is the implementation of the class: Robot learning 101 (JBNU 2025 fall, M.S course)

## Setup

Step1: Prepare dataset (I used QuickDraw dataset)

## Training

`python train.py --db_root "your db_root" --dbname "e.g. sunny" --use_reproj --smoothness --reproj_option OP`

## Eval

`python eval.py --restore_ckpt "your checkpoint root" --dbname "e.g. sunny" --save_misc`


## Results

<div align="center">
  <img src="./Figure/figure2.jpg" alt="framework" style="width:100%; height:auto;">
</div>