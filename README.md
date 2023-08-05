# smi_unet_prostate

Here is the official implementation of weakly supervised Unet.  We are supplementing this project. After the article is accepted, we will add more details, if you like our work, please give us a star!

Now I can give you some additional details of the project.

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/11599c5e-f9f3-4507-b413-4e25797db27d)


We are all aware of the challenges associated with systematic biopsy for detecting prostate cancer, as it lacks precision due to the unpredictable nature of cancer occurrence. While current medical imaging and targeted biopsy methods have improved the detection rate, systematic biopsy remains necessary for evaluating non-significant portions of the prostate.

Is there a way to make systematic biopsy as accurate and refined as targeted biopsy? Deep learning comes to mind, given its recent advancements. However, a major limitation is that even the most skilled radiologists struggle to accurately identify all prostate cancers, so how can we teach a machine to do so?

In a groundbreaking approach, a weakly supervised method was employed to learn from previously conducted systematic biopsies. This allows the model to uncover real correlations between MRI spatial information and pathology.

The overall process of establishing the model is outlined as follows:

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/1102a412-51fd-4d32-aea7-1da507b4f3b9)

We completed the model training and validation according to the following steps：First, the model would be trained to predict Gleason grade of biopsy (grade ≥ 7) in the training set, which allows the model to correlate spatial location and the probability of potential cancer. The trained model was used to generate a probability map of cancer distribution in the whole prostate based on MR images. Therefore, the model learns completely from the location and grade information of each biopsy, without any prior knowledge or misdirection of the radiologist. Then, to verify the performance of in searching for potential cancer areas, the model was used to generate 3D maps of potential cancer areas for each MR image, and the maps were evaluated with systematic biopsy in the testing set. 

Our specific model implementation is not complicated, which is due to our experimental finding that the simpler the model, the more robust it is.
 It can be represented as follows:

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/448a3a57-b924-4b78-897f-b1157dd8b5b1)

The backbone of it is a Unet which is obvious because it is a segmentation task. How do you do a supervision based on biopsy data? We can see a WS module. Let me give you a schematic diagram to understand it:

![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/dcf75dac-e942-4a8e-a807-1ea1a4e92ac8)

It can be considered as a multi-example task, where we interpolate each biopsy core to find each point, and then use maxpooling to complete multi-example learning. You can see model.py for a detailed implementation.


