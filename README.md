# Bare Struncture from Motion model

This is an implementation using pytorch of the basic approach of Structure from Motion models (or Self-supervised monocular depth estimation) models as explained [here (Unsupervised depth estimation)](https://notanymike.github.io/Unsupervised-depth-estimation/)

# Install

so far if you face any problem installing pytorch3d follow this quick instructions [here](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)

# Run

Run

     python train.py --height 320 --width 320 --dataset_folder /SSD/Documents/shapes3d/dataset_generated

for example. For more information run `python train.py --help` or see `options.py` for all the possible configurations.

## Tests

Run `py.test -s -v`
