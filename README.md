# smi_unet_prostate

Here is the official implementation of weakly supervised Unet. 

Now I can give you some additional details of the project.

![SceneView_1_2](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/b45c7d01-20aa-4f83-b95b-7ee22e4ac371)

We all know that the prostate is a cancer that requires systematic biopsy, which is extremely unfavorable for cancer detection, because the doctor does not know where the cancer will appear. Although current medical imaging and target-line biopsy methods have greatly improved the detection rate of prostate cancer, systematic biopsy remains the only method for evaluating the fraction of prostate cancer that is not significant.

So is there any way to make systematic biopsy as precise and elegant as target line biopsy? We think of deep learning, no doubt because of its breakthroughs in recent years. But a serious limitation is that if even the most expert radiologist cannot accurately map all prostate cancers, how can we teach a machine to do so?

In a groundbreaking attempt, a weakly supervised approach was used to fully learn knowledge from previously completed systematic biopsies. This gives the model the opportunity to find some real correlations between MRI spatial information and pathology.

The overall process of establishing the model is as follows:
![图片](https://github.com/Zhengyao0202/weakly_unet_prostate/assets/119670761/1102a412-51fd-4d32-aea7-1da507b4f3b9)







