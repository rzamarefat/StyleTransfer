# StyleTransfer
## A Common Implementation of StyleTransfer in Pytorch

### Sample 1
![StyleTransfer](https://user-images.githubusercontent.com/79300456/173231144-07160c7d-7b06-44d9-8cce-136fb9f6457a.jpg)
### Sample 2
![StyleTransfer2 (1)](https://user-images.githubusercontent.com/79300456/173231169-0cc65e6f-23f3-403d-ab97-b18d572cb7b7.jpg)

## How to use
1- Make an conda env
2- Install Pyotorch and torchvision
3- Go ahead and install the following packages
```
pip install tqdm opencv-python 
```

After installing all the dependencies run the following command:

```
python train.py --path-to-original-image path/to/original/image --path-to-style-image --path/to/style/image --path-to-save-output-image --path/to/save/output/image --num-steps 10000
```
Please refer to train.py module to see the other options. 

