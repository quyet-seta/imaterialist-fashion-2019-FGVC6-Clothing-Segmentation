# imaterialist-fashion-2019-FGVC6-Clothing-Segmentation
## Dataset
https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6
![image](https://user-images.githubusercontent.com/71560376/133889434-7c0e8c4a-e25d-4ce8-a84b-fc22409671bd.png)
## Dependences
1. Python 3.x
2. Numpy 1.19.5
3. Tensorflow 2.5.0
## Model
1. Pre-train MobilenetV2 Unet
![image](https://user-images.githubusercontent.com/71560376/133889501-063b335e-1d34-4e52-ad5b-80caba56de5f.png)
![image](https://user-images.githubusercontent.com/71560376/133889508-6141f341-e54a-4ba5-81ce-181171da8545.png)
![image](https://user-images.githubusercontent.com/71560376/133889482-4889c04d-b853-4bbf-a53a-bc2db1153fa1.png)
2. DeeplabV3 Pretrain Resnet50
![image](https://user-images.githubusercontent.com/71560376/133889581-a8de3f81-9ab1-4ab5-bc9f-3bb79a11a6e8.png)
3. Accuracy

MobinetV2-Unet  | Resnet50-DeeplabV3
------------- | -------------
86%  | 90%


