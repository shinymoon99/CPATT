#!/bin/bash

# Specify the total number of folds (n)
total_folds=5  # Replace 5 with the actual number of folds

# Loop through fold numbers from 1 to n
for ((fold_num=1; fold_num<=total_folds; fold_num++))
do
    echo "Running inference for fold $fold_num..."

    # Run train.py with dataset location and current fold number as arguments
    python ./code/ECI/inferenceESC.py --dataset_loc ./data/ESC_n_fold --fold_num $fold_num

    echo "inference for fold $fold_num completed."
done
