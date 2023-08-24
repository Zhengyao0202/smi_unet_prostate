

from monai.transforms import (
    CastToTyped,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    LoadImage,
    NormalizeIntensity,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandZoomd,
    SpatialCrop,
    SpatialPadd,
    ToTensord,
    EnsureTyped,
    AsChannelFirstd,
    AddChanneld,
    Spacingd,
EnsureChannelFirstd,
NormalizeIntensityd,
CenterSpatialCropd,
Orientationd,
SpatialPadd
)
import torch
from monai.transforms.compose import MapTransform




class gernerate_grid(MapTransform):

    def __call__(self, data):


        d = dict(data)
        # print(d.keys())

        line=d["line_biopsy"]
        core=d["Core Label"]

        img_affine=d["ProstateSurface"].affine.inverse().to(torch.float32)
        A=img_affine[:3,:3]
        B=img_affine[:3,3:4]

        p = torch.zeros([24 , len(line[0][0]) , 3])

        for i in  range(len(line)):
            temp=torch.tensor(line[i])
            temp2=torch.mm(A,temp)+B
            temp3=temp2.T
            p[i]=temp3/torch.tensor([192,192,48])



        d['grid_sample']=p.float()
        d["MR_location"] = d["MR_location"]*d["ProstateSurface"]
        d["ProstateSurface"] = d["ProstateSurface"] * d['label_PSA']
        d["MR_location"] = d["MR_location"].float()
        d['label_proportion'] = torch.tensor(d['label_proportion']).float()

        return dict([(key,d[key]) for key in ['name',"MR_location", "ProstateSurface",'target',\
                                              'grid_sample','mask','target_mask','system_mask','label_Gleason', 'label_proportion']])


def  get_transform(mode='train'):


    keys = ["MR_location", "ProstateSurface", 'target']

    train_transform = [
        # biopsy_unnest(keys=["None"]),
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=[0.6640625, 0.6640625, 1.5], mode=["bilinear", "nearest", "nearest"]),
        # Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
        NormalizeIntensityd(keys=["MR_location"], nonzero=False),
        CenterSpatialCropd(keys=keys, roi_size=[192, 192, 48]),
        SpatialPadd(keys=keys,  spatial_size=[192, 192, 48]),

        # LoadImaged(keys=["MR_location", "ProstateSurface",'target1','target2','target3'],allow_missing_keys=True),
        # target_process(keys=["MR_location", "ProstateSurface",'target1','target2','target3']),
        RandFlipd(keys=keys, spatial_axis=[0], prob=0.5),
        RandFlipd(keys=keys, spatial_axis=[1], prob=0.5),
        RandFlipd(keys=keys, spatial_axis=[2], prob=0.5),

        gernerate_grid(keys=keys),
        EnsureTyped(keys=[ "MR_location", "ProstateSurface", 'target', \
                          'grid_sample', 'mask', 'target_mask', 'system_mask', 'label_Gleason', 'label_proportion']),


    ]



    val_transform = [
        # biopsy_unnest(keys=["None"]),
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=[0.6640625, 0.6640625, 1.5], mode=["bilinear", "nearest", "nearest"]),
        # Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
        NormalizeIntensityd(keys=["MR_location"], nonzero=False),
        CenterSpatialCropd(keys=keys, roi_size=[192, 192, 48]),
        SpatialPadd(keys=keys, spatial_size=[192, 192, 48]),

        # LoadImaged(keys=["MR_location", "ProstateSurface",'target1','target2','target3'],allow_missing_keys=True),
        # target_process(keys=["MR_location", "ProstateSurface",'target1','target2','target3']),
        gernerate_grid(keys=keys),
        EnsureTyped(keys=[ "MR_location", "ProstateSurface", 'target', \
                          'grid_sample', 'mask', 'target_mask', 'system_mask', 'label_Gleason', 'label_proportion']),

    ]






    if mode =='train':
        temp=Compose(train_transform)
        return  temp

    elif mode =='val':
        temp = Compose(val_transform)
        return temp



