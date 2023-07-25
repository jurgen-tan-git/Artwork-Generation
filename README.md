# CycleGAN implementation in Pytorch

This is a Pytorch implementation of CycleGAN to transfer Van Gogh's art style onto pictures.
To download the dataset, run ./download.sh in your terminal before executing main.py. Please install the required packages using 
```pip3 install -r requirements.txt```

To compute the FID score between two datasets, where images of each dataset are contained in an individual folder:
```
python -m pytorch_fid path/to/dataset1 path/to/dataset2
```

To run the evaluation on GPU, use the flag `--device cuda:N`, where `N` is the index of the GPU to use.