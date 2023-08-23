# smi_unet_prostate

Here is the official implementation of weakly supervised Unet.  We are supplementing this project. After the article is accepted, we will add more details, if you like our work, please give us a star!

Now I can give you some additional details of the project.

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/11599c5e-f9f3-4507-b413-4e25797db27d)


We are all aware of the challenges associated with systematic biopsy for detecting prostate cancer, as it lacks precision due to the unpredictable nature of cancer occurrence. While current medical imaging and targeted biopsy methods have improved the detection rate, systematic biopsy remains necessary for evaluating non-significant portions of the prostate.

Is there a way to make systematic biopsy as accurate and refined as targeted biopsy? Deep learning comes to mind, given its recent advancements. However, a major limitation is that even the most skilled radiologists struggle to accurately identify all prostate cancers, so how can we teach a machine to do so?


# model

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

# data

Here, we must thank the open source database:

[1]	Natarajan S, Priester A, Margolis D, Huang J, Marks L. Prostate MRI and ultrasound with pathology and coordinates of tracked biopsy (Prostate-MRI-US-Biopsy). Cancer Imaging Arch. 2020;10:7937. DOI: 10.7937/TCIA.2020.A61IOC1A

 ![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/5da70c53-c07c-44b5-83e7-acbb538f5a11)

This study utilized a publicly available dataset, namely the Prostate MRI and ultrasound with pathology and coordinates of tracked biopsy (Prostate-MRI-US-Biopsy) dataset, obtained from The Cancer Imaging Archive (TCIA). The dataset was derived from tracked biopsy sessions using the Artemis biopsy system, which involved the fusion of real-time ultrasound and preoperative MRI images. Biopsy cores were sampled from regions of interest identified on MRI scans. Additionally, systematic biopsy cores were also sampled using a 12-core digital template. The Artemis system recorded the locations of targeted and systematic cores relative to the ultrasound scan, as well as the MRI biopsy coordinates for most cases.

The dataset includes STL files and biopsy overlays that can be visualized in 3D Slicer with the SlicerHeart extension. Furthermore, spreadsheets summarizing biopsy and MRI target data are available. Detailed information about the dataset can be found in the Cancer Imaging Archive ( https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=68550661. )
MRI targets in this dataset were defined using multiparametric MRI, such as T2-weighted, diffusion-weighted, and perfusion-weighted sequences. These targets were scored on a Likert-like scale with close correspondence to PIRADS version 2. T2-weighted MRI was used to trace ROI contours, and it is the only sequence provided in this dataset. MR imaging was performed on Siemens scanners including 3 Tesla Trio, Verio, or Skyra.

Ultrasound scans were performed using Hitachi Hi-Vision 5500 7.5 MHz or Noblus C41V 2-10 MHz end-fire probe. 3D scans were acquired by rotating the end-fire probe 200 degrees about its axis, and the volume was resampled with isotropic resolution.

The dataset consists of patients who were suspected of having prostate cancer due to elevated PSA levels and/or suspicious imaging findings. Patients who underwent or were scheduled to undergo routine, standard-of-care prostate biopsy at the UCLA Clark Urology Center were included in the study.

Of the 1151 patients in the dataset, 777 patients met the inclusion and exclusion criteria, which is shown in Supplementary Fig. 1. The patients were randomly divided into a subset used for training (n = 600) and a separate testing set (n = 177). 

# experiment

1. preprocess

   

