#!/bin/bash

task=sementic_parsing_dual
exp_path=exp


# paras for model

batchSize=4
test_batchSize=128
optim=adam
lr=0.001
dropout=0.5
max_norm=5
l2=1e-5
max_epoch=100
deviceId=0

python3 ./scripts/lstm.py --task $task --experiment $exp_path \
        --max_epoch $max_epoch --batchSize $batchSize \
        --optim $optim --lr $lr --l2 $l2 --dropout $dropout\
         --max_norm $max_norm --test_batchSize $test_batchSize \
