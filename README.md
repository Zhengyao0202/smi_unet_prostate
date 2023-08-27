
# Detecting prostate cancer with obscure MRI appearance using a weakly supervised deep learning model 


![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/85fee6ab-7e0f-406a-b41c-8a04729f6bb9)

![Static Badge](https://img.shields.io/badge/model-WSUNet-blue)   ![Static Badge](https://img.shields.io/badge/Phase%20-Model%20testing%20-red)  ![Static Badge](https://img.shields.io/badge/Field%20-%20Prostate%20-%20Green)  ![Static Badge](https://img.shields.io/badge/Project%20-%20under%20development%20-%20orange)




## overview

**Here is the official implementation of weakly supervised Unet (WSUNet).** 

**If you like our work, please give us a star!**

## Prerequisites
* Python 3.8.1 
* Pytorch 1.1.0 and torchvision (https://pytorch.org/)
* MONAI 1.10
* CUDA 10.0.130, CuDNN 7.3, and Ubuntu 18.04.
* device: 1 A100 GPU or 2 V100 GPU

## Preparation of data
Here, we used an open source database:

[1]	Natarajan S, Priester A, Margolis D, Huang J, Marks L. Prostate MRI and ultrasound with pathology and coordinates of tracked biopsy (Prostate-MRI-US-Biopsy). Cancer Imaging Arch. 2020;10:7937. DOI: 10.7937/TCIA.2020.A61IOC1A

The data were collated by different MRI detections and it was collated into json format. 

we provides a sample  `data_one_sample.json` for your reference:

```Shell
"Core Label": [
      "TARGET OR PRIOR POSITIVE",
      "TARGET OR PRIOR POSITIVE",
      "TARGET OR PRIOR POSITIVE"
    ],
    "MR_image_base": [
      [
        113.46211948529411,
        151.75269292279413,
    .......

```

## train and test model 

train a model:

```Shell
python  main.py
--data_train [your train data]
--data_test  [your test data]
--log_dir  [output dir]
--batch_size 12
--max_epoches 100
```








## more detail about this project 


### introduction
Now I can give you some additional details of the project.

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/11599c5e-f9f3-4507-b413-4e25797db27d)


We are all aware of the challenges associated with systematic biopsy for detecting prostate cancer, as it lacks precision due to the unpredictable nature of cancer occurrence. While current medical imaging and targeted biopsy methods have improved the detection rate, systematic biopsy remains necessary for evaluating non-significant portions of the prostate.

Is there a way to make systematic biopsy as accurate and refined as targeted biopsy? Deep learning comes to mind, given its recent advancements. However, a major limitation is that even the most skilled radiologists struggle to accurately identify all prostate cancers, so how can we teach a machine to do so?


### model

In a groundbreaking approach, a weakly supervised method was employed to learn from previously conducted systematic biopsies. This allows the model to uncover real correlations between MRI spatial information and pathology.

The overall process of establishing the model is outlined as follows:

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/1102a412-51fd-4d32-aea7-1da507b4f3b9)

We completed the model training and validation according to the following steps：First, the model would be trained to predict Gleason grade of biopsy (grade ≥ 7) in the training set, which allows the model to correlate spatial location and the probability of potential cancer. The trained model was used to generate a probability map of cancer distribution in the whole prostate based on MR images. Therefore, the model learns completely from the location and grade information of each biopsy, without any prior knowledge or misdirection of the radiologist. Then, to verify the performance of in searching for potential cancer areas, the model was used to generate 3D maps of potential cancer areas for each MR image, and the maps were evaluated with systematic biopsy in the testing set. 

Our specific model implementation is not complicated, which is due to our experimental finding that the simpler the model, the more robust it is.
 It can be represented as follows:

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/448a3a57-b924-4b78-897f-b1157dd8b5b1)

The backbone of it is a Unet which is obvious because it is a segmentation task. How do you do a supervision based on biopsy data? We can see a WS module. Let me give you a schematic diagram to understand it:

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/dcf75dac-e942-4a8e-a807-1ea1a4e92ac8)

It can be considered as a multi-instance task, where we interpolate each biopsy core to find each point, and then use maxpooling to complete multi-instance learning. You can see model.py for a detailed implementation.

### data

Here, we used an open source database:

[1]	Natarajan S, Priester A, Margolis D, Huang J, Marks L. Prostate MRI and ultrasound with pathology and coordinates of tracked biopsy (Prostate-MRI-US-Biopsy). Cancer Imaging Arch. 2020;10:7937. DOI: 10.7937/TCIA.2020.A61IOC1A

 ![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/5da70c53-c07c-44b5-83e7-acbb538f5a11)

The dataset includes STL files and biopsy overlays that can be visualized in 3D Slicer with the SlicerHeart extension. Furthermore, spreadsheets summarizing biopsy and MRI target data are available. Detailed information about the dataset can be found in the Cancer Imaging Archive ( https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=68550661. )

The dataset consists of patients who were suspected of having prostate cancer due to elevated PSA levels and/or suspicious imaging findings. Patients who underwent or were scheduled to undergo routine, standard-of-care prostate biopsy at the UCLA Clark Urology Center were included in the study.

Of the 1151 patients in the dataset, 777 patients met the inclusion and exclusion criteria, which is shown belew. The patients were randomly divided into a subset used for training (n = 600) and a separate testing set (n = 177). 

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/f3915561-e391-42dc-afae-bd22385a4054)







