#!/bin/bash

# Specify the total number of folds (n)
total_folds=10  # Replace 5 with the actual number of folds

# Loop through fold numbers from 1 to n
for ((fold_num=1; fold_num<=total_folds; fold_num++))
do
    echo "Running training for fold $fold_num..."

    # Run train.py with dataset location and current fold number as arguments
    python ./code/ECI/train.py --dataset_loc ./data/CTB_n_fold --fold_num $fold_num

    echo "Training for fold $fold_num completed."
done
