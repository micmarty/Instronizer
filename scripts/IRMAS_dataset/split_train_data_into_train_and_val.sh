#!/bin/bash

##
# Distributes spectrograms from one dataset into validation 
## by Michał Martyniak

# Set working directory and get input path from script call
ROOT_DIR=$1

# IRMAS spectrograms -> 6705
# val -> 15%
num_val=1005

# 1005 / 11 instruments = 91
# PNG files per dataset
n_val=91

# For every class folder (without root)
for class_path in $(find $ROOT_DIR -links 2 -type d)
do	
	# Randomly choose n files and move them to a new dataset folder
	val_samples=$(find $class_path -links 1 | shuf -n $n_val)
	mkdir --parents val/$(basename $class_path)
	mv $val_samples --target-directory="val/$(basename $class_path)"
done
	
