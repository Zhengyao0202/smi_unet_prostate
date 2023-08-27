
from monai.data.utils import pad_list_data_collate

import torch
from pytool import make_dir,EarlyStopping, divide_data
from transforms2 import get_transform
from my_model import SmiUnet
from monai.transforms import Activations, AsDiscrete, Compose
from monai.metrics import ROCAUCMetric
from monai.data import (
    Dataset,
    )
import json

import torch
import numpy as np


train_transform = get_transform(mode='train')
val_transform = get_transform(mode='val')

post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
post_pre_pred = Compose([Activations(sigmoid=True)])

auc_metric_0 = ROCAUCMetric()
auc_metric_sys = ROCAUCMetric()
auc_metric_sys_high_3 = ROCAUCMetric()
auc_metric_sys_low_3 = ROCAUCMetric()
auc_metric_3 = ROCAUCMetric()
auc_metric_4 = ROCAUCMetric()
auc_metric_5 = ROCAUCMetric()


auc_metric_tar = ROCAUCMetric()
single_auc_metric_sys=ROCAUCMetric()

affine_par = True

from monai.data import (
    DataLoader,
)
import os
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):

    data_test_path = args.data_test_path
    ori_out_dir =args.ori_out_dir


    model=SmiUnet().cuda()

    with open(data_test_path, "r") as f:
        test_data = json.load(f)





    testset = Dataset(
        data=test_data,
        transform=val_transform
    )





    test_loader = DataLoader(testset,
                             batch_size=1,
                             num_workers=4,
                             pin_memory=torch.cuda.is_available(),
                             shuffle=False,
                             collate_fn=pad_list_data_collate
                             )


    # i=0
    all_meta=[]


    index=['target1','target2','target3','target4','target5']

    for i in range(len(test_data)):
        meta={}
        temp_pirads=0
        for key in index:
            if key in test_data[i]:
                now_pirads=test_data[i][key+'_pirads']
                if now_pirads>temp_pirads:
                    temp_pirads = now_pirads

        meta['PIRADS']=temp_pirads


        temp_PSA=0
        if test_data[i]['PSA'] > temp_PSA:
            temp_PSA = test_data[i]['PSA']

        meta['PSA'] = temp_PSA

        all_meta.append(meta)



    model.load_state_dict(torch.load(ori_out_dir + "best_metric_model.pth"))
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32).cuda()
        y = torch.tensor([], dtype=torch.float32).cuda()

        y_pred_sys = torch.tensor([], dtype=torch.float32).cuda()
        y_sys = torch.tensor([], dtype=torch.float32).cuda()

        y_pred_sys_high_3 = torch.tensor([], dtype=torch.float32).cuda()
        y_sys_high_3 = torch.tensor([], dtype=torch.float32).cuda()

        y_pred_sys_low_3 = torch.tensor([], dtype=torch.float32).cuda()
        y_sys_low_3 = torch.tensor([], dtype=torch.float32).cuda()

        y_pred_sys_3 = torch.tensor([], dtype=torch.float32).cuda()
        y_sys_3 = torch.tensor([], dtype=torch.float32).cuda()

        y_pred_sys_4 = torch.tensor([], dtype=torch.float32).cuda()
        y_sys_4 = torch.tensor([], dtype=torch.float32).cuda()


        y_pred_sys_5 = torch.tensor([], dtype=torch.float32).cuda()
        y_sys_5 = torch.tensor([], dtype=torch.float32).cuda()







        y_pred_tar = torch.tensor([], dtype=torch.float32).cuda()
        y_tar = torch.tensor([], dtype=torch.float32).cuda()
    i=0
    for batch in test_loader:


        meta=all_meta[i]
        MR, ProstateSurface, target, grid_sample, mask, label_Gleason, target_mask, system_mask = \
            batch["MR_location"].cuda(), \
            batch["ProstateSurface"].cuda(), \
            batch['target'].cuda(), \
            batch['grid_sample'].cuda(), \
            batch['mask'].cuda(), \
            batch['label_Gleason'].cuda(), \
            batch['target_mask'].cuda(), \
            batch['system_mask'].cuda(),

        img = torch.concat([MR, ProstateSurface, target], dim=1)


        with torch.no_grad():
            biopsy_all = model(img, grid_sample).float()
            pro=model.origin_forward(img)




        outputs = torch.masked_select(biopsy_all, mask)
        val_label = torch.masked_select(label_Gleason, mask).float()

        target_outputs = torch.masked_select(biopsy_all, target_mask)
        target_val_label = torch.masked_select(label_Gleason, target_mask).float()

        system_outputs = torch.masked_select(biopsy_all, system_mask)
        system_val_label = torch.masked_select(label_Gleason, system_mask).float()




        meta['i'] = i
        meta['name'] = batch['name'][0]
        meta['num_mask'] = mask.cpu().sum().item()
        meta['num_target_mask'] = target_mask.cpu().sum().item()
        meta['num_system_mask'] = system_mask.cpu().sum().item()
        meta['num_target_mask_gleason']=target_val_label.cpu().sum().item()
        meta['num_system_mask_gleason'] = system_val_label.cpu().sum().item()


        y_pred_act_single = post_pre_pred(system_outputs)
        if system_val_label.sum().item()>0 and system_val_label.sum().item()<system_val_label.shape[0]:


            single_auc_metric_sys(y_pred_act_single, system_val_label)
            auc_single= single_auc_metric_sys.aggregate()
            single_auc_metric_sys.reset()
            meta['system_AUC'] =  auc_single

        else :
            meta['system_AUC'] = None


        meta['y_pred_act_single'] = y_pred_act_single.cpu().numpy().tolist()
        meta['system_val_label'] = system_val_label.cpu().numpy().tolist()

        # all_meta.append(meta)
        y_pred = torch.cat([y_pred, outputs], dim=0)
        # val_label = torch.stack([post_label(i) for i in val_label], dim=0)
        y = torch.cat([y, val_label], dim=0)

        y_pred_tar = torch.cat([y_pred_tar, target_outputs], dim=0)
        # val_label = torch.stack([post_label(i) for i in val_label], dim=0)
        y_tar = torch.cat([y_tar, target_val_label], dim=0)

        y_pred_sys = torch.cat([y_pred_sys, system_outputs], dim=0)
        y_sys = torch.cat([y_sys, system_val_label], dim=0)

        if meta['PIRADS'] >=3:
            y_pred_sys_high_3 = torch.cat([y_pred_sys_high_3, system_outputs], dim=0)
            y_sys_high_3 = torch.cat([y_sys_high_3, system_val_label], dim=0)


            if meta['PIRADS']==3:
                y_pred_sys_3 = torch.cat([y_pred_sys_3, system_outputs], dim=0)
                y_sys_3 = torch.cat([y_sys_3, system_val_label], dim=0)
            elif meta['PIRADS']==4:
                y_pred_sys_4 = torch.cat([y_pred_sys_4, system_outputs], dim=0)
                y_sys_4 = torch.cat([y_sys_4, system_val_label], dim=0)
            elif meta['PIRADS'] == 5:
                y_pred_sys_5 = torch.cat([y_pred_sys_5, system_outputs], dim=0)
                y_sys_5 = torch.cat([y_sys_5, system_val_label], dim=0)


        elif meta['PIRADS'] <3:
            y_pred_sys_low_3 = torch.cat([y_pred_sys_low_3, system_outputs], dim=0)
            y_sys_low_3 = torch.cat([y_sys_low_3, system_val_label], dim=0)






        print(i,batch['name'])


        del MR, ProstateSurface, target, grid_sample, mask, label_Gleason, target_mask, system_mask

        i=i+1

    acc_value_0 = torch.eq(post_pred(y_pred), y)
    acc_metric_0 = acc_value_0.sum().item() / len(acc_value_0)

    acc_value_tar = torch.eq(post_pred(y_pred_tar), y_tar)
    acc_metric_tar = acc_value_tar.sum().item() / len(acc_value_tar)

    acc_value_sys = torch.eq(post_pred(y_pred_sys), y_sys)
    acc_metric_sys = acc_value_sys.sum().item() / len(acc_value_sys)

    acc_value_sys_high_3 = torch.eq(post_pred(y_pred_sys_high_3), y_sys_high_3)
    acc_metric_sys_high_3 = acc_value_sys_high_3.sum().item() / len(acc_value_sys_high_3)

    acc_value_sys_low_3 = torch.eq(post_pred(y_pred_sys_low_3), y_sys_low_3)
    acc_metric_sys_low_3 = acc_value_sys_low_3.sum().item() / len(acc_value_sys_low_3)





    y_pred_act_0 = post_pre_pred(y_pred)
    auc_metric_0(y_pred_act_0, y)
    auc_result_0 = auc_metric_0.aggregate()
    auc_metric_0.reset()

    y_pred_act_tar = post_pre_pred(y_pred_tar)
    auc_metric_tar(y_pred_act_tar, y_tar)
    auc_result_tar = auc_metric_tar.aggregate()
    auc_metric_tar.reset()

    y_pred_act_sys = post_pre_pred(y_pred_sys)
    auc_metric_sys(y_pred_act_sys, y_sys)
    auc_result_sys = auc_metric_sys.aggregate()
    auc_metric_sys.reset()

    y_pred_act_sys_high_3 = post_pre_pred(y_pred_sys_high_3)
    auc_metric_sys_high_3(y_pred_act_sys_high_3, y_sys_high_3)
    auc_result_sys_high_3 = auc_metric_sys_high_3.aggregate()
    auc_metric_sys_high_3.reset()


    y_pred_act_sys_low_3 = post_pre_pred(y_pred_sys_low_3)
    auc_metric_sys_low_3(y_pred_act_sys_low_3, y_sys_low_3)
    auc_result_sys_low_3 = auc_metric_sys_low_3.aggregate()
    auc_metric_sys_low_3.reset()

    y_pred_act_3 = post_pre_pred(y_pred_sys_3)
    auc_metric_3(y_pred_act_3, y_sys_3)
    auc_result_3 = auc_metric_3.aggregate()
    auc_metric_3.reset()

    y_pred_act_4 = post_pre_pred(y_pred_sys_4)
    auc_metric_4(y_pred_act_4, y_sys_4)
    auc_result_4 = auc_metric_4.aggregate()
    auc_metric_4.reset()

    y_pred_act_5 = post_pre_pred(y_pred_sys_5)
    auc_metric_5(y_pred_act_5, y_sys_5)
    auc_result_5 = auc_metric_5.aggregate()
    auc_metric_5.reset()








    test_other_output =  ori_out_dir+'test_save/'

    make_dir(test_other_output)
    np.save(test_other_output+ 'y_sys.npy', y_sys.cpu().numpy())

    np.save(test_other_output + 'y_pred_act_sys.npy', y_pred_act_sys.cpu().numpy())

    np.save(test_other_output+ 'y_sys_high_3.npy', y_sys_high_3.cpu().numpy())
    np.save(test_other_output + 'y_pred_act_sys_high_3.npy', y_pred_act_sys_high_3.cpu().numpy())

    np.save(test_other_output+ 'y_sys_low_3.npy', y_sys_low_3.cpu().numpy())
    np.save(test_other_output + 'y_pred_act_sys_low_3.npy', y_pred_act_sys_low_3.cpu().numpy())

    np.save(test_other_output+ 'y_sys_3.npy', y_sys_3.cpu().numpy())
    np.save(test_other_output + 'y_pred_act_3.npy', y_pred_act_3.cpu().numpy())

    np.save(test_other_output+ 'y_sys_4.npy', y_sys_4.cpu().numpy())
    np.save(test_other_output + 'y_pred_act_4.npy', y_pred_act_4.cpu().numpy())

    np.save(test_other_output+ 'y_sys_5.npy', y_sys_5.cpu().numpy())
    np.save(test_other_output + 'y_pred_act_5.npy', y_pred_act_5.cpu().numpy())



    print('auc_result_0:',auc_result_0 )
    print('auc_result_tar:',auc_result_tar )
    print('auc_result_sys:',auc_result_sys )
    print('auc_result_sys_high_3:',auc_result_sys_high_3 )
    print('auc_result_sys_low_3:',auc_result_sys_low_3 )
    print('auc_result_3:',auc_result_3 )
    print('auc_result_4:',auc_result_4 )
    print('auc_result_5:',auc_result_5 )


    output_dir=ori_out_dir+'statistics/'
    make_dir(output_dir)



    with open(output_dir+'all_test.json', "w") as f:
        json.dump(all_meta,  # 待写入数据
                  f,  # File对象
                  indent=2,  # 空格缩进符，写入多行
                  sort_keys=True)  # 键的排序)
        f.close()



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument("--data_test_path", default='data_record/divided_data1/train.json', type=str)
    parser.add_argument("--ori_out_dir", default='result_record/divided_data1/', type=str)


    args = parser.parse_args()
    print(args)

    main(args)

