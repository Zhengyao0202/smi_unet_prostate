# smi_unet_prostate

Here is the official implementation of weakly supervised Unet. 

Now I can give you some additional details of the project.

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/11599c5e-f9f3-4507-b413-4e25797db27d)


We all know that the prostate is a cancer that requires systematic biopsy, which is extremely unfavorable for cancer detection, because the doctor does not know where the cancer will appear. Although current medical imaging and target-line biopsy methods have greatly improved the detection rate of prostate cancer, systematic biopsy remains the only method for evaluating the fraction of prostate cancer that is not significant.

So is there any way to make systematic biopsy as precise and elegant as target line biopsy? We think of deep learning, no doubt because of its breakthroughs in recent years. But a serious limitation is that if even the most expert radiologist cannot accurately map all prostate cancers, how can we teach a machine to do so?

In a groundbreaking attempt, a weakly supervised approach was used to fully learn knowledge from previously completed systematic biopsies. This gives the model the opportunity to find some real correlations between MRI spatial information and pathology.

The overall process of establishing the model is as follows:

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/1102a412-51fd-4d32-aea7-1da507b4f3b9)

We completed the model training and validation according to the following steps：First, the model would be trained to predict Gleason grade of biopsy (grade ≥ 7) in the training set, which allows the model to correlate spatial location and the probability of potential cancer. The trained model was used to generate a probability map of cancer distribution in the whole prostate based on MR images. Therefore, the model learns completely from the location and grade information of each biopsy, without any prior knowledge or misdirection of the radiologist. Then, to verify the performance of in searching for potential cancer areas, the model was used to generate 3D maps of potential cancer areas for each MR image, and the maps were evaluated with systematic biopsy in the testing set. 

Our specific model implementation is not complicated, which is due to our experimental finding that the simpler the model, the more robust it is.
 It can be represented as follows:

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/448a3a57-b924-4b78-897f-b1157dd8b5b1)

The backbone of it is a Unet which is obvious because it is a segmentation task. How do you do a supervision based on biopsy data? We can see a WS module. Let me give you a schematic diagram to understand it:

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/dcf75dac-e942-4a8e-a807-1ea1a4e92ac8)

It can be considered as a multi-example task, where we interpolate each biopsy core to find each point, and then use maxpooling to complete multi-example learning. You can see model.py for a detailed implementation.


