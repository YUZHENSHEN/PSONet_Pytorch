This is the procedure for implementing PSONet.


enviroment.yml-----------------Environment setup file for running PSONet

main.py-----------------Main function
PSONet.py-----------------PSONet Framework
lcnn.py---------------------------The designed lightweight CNN in PSONet
sampleselect.py------------------Function for selecting samples
function_sup.py------------------Other auxiliary functions

data- -----------------------------Path to save experimental data
    I1.mat and I2.mat-----------------------Bitemporal image after normalization
    gt.mat------------------------------Change reference map
    index.mat---------------------------Pixel indexes needed to calculated detection accuracy
deep feature of DCVA------------Path to save deep feature from DCVA 
    deepfea1.mat and deepfea2.mat-----------------------Deep features of the bi-temporal images I1 and I2 extracted by DCVA
    (The deepfea1.mat and deepfea2.mat of the ZY3 dataset in the open code can be downloaded from: 
    https://drive.google.com/drive/folders/1sjEnpI1l39g1jX7xYusFiPJPpdAp9S2o?usp=sharing)
result-----------------------------Path to save change detection result
    ACCURACY.txt-----------------Record detection accuracy and runtime
    BCDM.mat---------------------Change detection results（.mat）
    BCDM.png---------------------Change detection results（.png）
    WTSCS.mat-----------------Initial“weak-to-strong” change signal image
    PSONet_1.pth-PSONet_5.pth--Models trained in the first five progressions
    PSONet_6.pth------------------Same as PSONet_5.pth

If you find our work useful for your research, please consider citing our paper:
@article{SHEN2025104792,
title = {Progressive Self-Optimization Network: An unsupervised change detection method for VHR optical remote sensing imagery},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {143},
pages = {104792},
year = {2025},
issn = {1569-8432},
doi = {https://doi.org/10.1016/j.jag.2025.104792},
url = {https://www.sciencedirect.com/science/article/pii/S156984322500439X},
author = {Yuzhen Shen and Francesca Bovolo and Yuchun Wei and Xudong Rui},
keywords = {Convolutional neural network, Progressive self-optimization, Unsupervised change detection, VHR remote sensing imagery}
}

If you have any questions, please contact yuzhenshen@nnu.edu.cn.
