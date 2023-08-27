from torch.utils.data import DataLoader
from monai.data import DataLoader
from monai.utils import set_determinism
import argparse
from monai.data.utils import pad_list_data_collate
import torch

from tool.pytool import make_dir,EarlyStopping, divide_data
from tool.metric import  get_result,get_result2

from transforms2 import get_transform
from monai.transforms import Activations, AsDiscrete, Compose, EnsureType
from monai.metrics import ROCAUCMetric
from monai.data import (
    Dataset,

    )
import json
from sklearn.model_selection import train_test_split, KFold

import torch
import numpy as np
import torch.nn as nn

affine_par = True


torch.multiprocessing.set_sharing_strategy('file_system')
import os
import random
import os
def setup_seed(seed):
   torch.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   set_determinism(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True

setup_seed(0)


set_determinism(0)


def main(args):


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    ori_out_dir=args.log_dir
    k_fold = 0
    train_log_dir=ori_out_dir+'train_output.txt'
    out_dir=ori_out_dir

    make_dir(ori_out_dir)

    with open(train_log_dir, "w") as f:
        json_str = json.dumps(vars(args), indent=0)
        f.write(json_str)
        f.write('\n')
        f.close()


    # device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_pre_pred = Compose([Activations(sigmoid=True)])


    train_transform = get_transform(mode='train')
    val_transform = get_transform(mode='val')

    # divide_data(ori_out_dir, all_files, label, args.seed)
    make_dir(out_dir + 'result_save/')

    with open(args.data_train, "r") as f:
        train_data = json.load(f)

    train_files, val_files = train_test_split(train_data, test_size=0.1,
                                              random_state=args.val_seed)

    trainset = Dataset(
        data=train_files,
        transform=train_transform
    )



    with open(args.data_test, "r") as f:
        test_data = json.load(f)

    testset = Dataset(
        data=test_data,
        transform=val_transform
    )


    
    valset = Dataset(
        data=val_files,
        transform=val_transform
    )


    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available(),
                              collate_fn=pad_list_data_collate,
                              # persistent_workers=False
                              )
    # create a validation data loader

    val_loader = DataLoader(valset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available(),
                            collate_fn=pad_list_data_collate
                            )


    test_loader = DataLoader(testset,
                            batch_size=1,
                            num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available(),
                            collate_fn=pad_list_data_collate
                            )






    from my_model import SmiUnet_wildcat

    model=SmiUnet_wildcat()


    # model = nn.DataParallel(model)
    model = model.cuda()


    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))

    optimizer = torch.optim.AdamW(model.parameters(), args.lr,)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoches)
    auc_metric_0 = ROCAUCMetric()
    auc_metric_sys = ROCAUCMetric()
    auc_metric_tar = ROCAUCMetric()


    early_stopping = EarlyStopping(patience=args.patience, verbose=True,path=ori_out_dir+"/best_metric_model.pth")


    # start a typical PyTorch training
    val_interval = args.val_interval
    save_interval=args.val_interval*2
    best_metric = -1
    best_metric_epoch = -1


    for epoch in range(args.max_epoches):
        print("-" * 20)
        print(f"epoch {epoch + 1}/{args.max_epoches}")
        model.train()
        epoch_loss = 0
        step = 0
        y_pred = torch.tensor([], dtype=torch.float32).cuda()
        y = torch.tensor([], dtype=torch.float32).cuda()

        for batch in train_loader:
            step += 1
            MR, ProstateSurface, target,  grid_sample, mask, label_Gleason, proportion = \
                batch["MR_location"].cuda(), \
                batch["ProstateSurface"].cuda(), \
                batch['target'].cuda(), \
                batch['grid_sample'].cuda(), \
                batch['mask'].cuda(), \
                batch['label_Gleason'].cuda(),\
                batch['label_proportion'].cuda()


            img = torch.concat([MR, ProstateSurface, target], dim=1)

            # biopsy_all = model(img, PSA, grid_sample, mask).float()
            biopsy_all = model(img, grid_sample, proportion).float()

            out = torch.masked_select(biopsy_all, mask).float()

            labels = torch.masked_select(label_Gleason, mask).float()

            loss = loss_function(out, labels)

            optimizer.zero_grad()
            # print('adc',adc.shape)
            # print('t2', t2.shape)
            # print('location', location.shape)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(trainset) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            y_pred = torch.cat([y_pred, out], dim=0)
            y = torch.cat([y, labels], dim=0)

        lr_scheduler.step()
        acc = torch.eq(post_pred(y_pred), y)
        acc = acc.sum().item() / len(acc)
        # acc_metric_1 = acc_value_1.sum().item() / len(acc_value_1)

        # print(f" fold{k_fold}   epoch {epoch + 1} average train acc: {acc_metric:.4f}")
        print(f" fold{k_fold}   epoch {epoch + 1}  train acc_value_0_1: {acc:.4f}  ")

        epoch_loss /= step
        print(f" fold{k_fold}   epoch {epoch + 1} average loss: {epoch_loss:.4f}")


        with open(train_log_dir, "a") as f:
            f.write(f"epoch {epoch + 1}    "
                    f"train acc_value_0: {acc:.4f}     "  
                    f"train average loss: {epoch_loss:.4f}\n" )
            f.close()


        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32).cuda()
                y = torch.tensor([], dtype=torch.float32).cuda()

                y_pred_sys = torch.tensor([], dtype=torch.float32).cuda()
                y_sys = torch.tensor([], dtype=torch.float32).cuda()

                y_pred_tar = torch.tensor([], dtype=torch.float32).cuda()
                y_tar = torch.tensor([], dtype=torch.float32).cuda()

                for batch in val_loader:

                    MR, ProstateSurface, target, grid_sample, mask, label_Gleason,target_mask,system_mask,proportion  = \
                        batch["MR_location"].cuda(), \
                        batch["ProstateSurface"].cuda(), \
                        batch['target'].cuda(), \
                        batch['grid_sample'].cuda(), \
                        batch['mask'].cuda(), \
                        batch['label_Gleason'].cuda(), \
                        batch['target_mask'].cuda(), \
                        batch['system_mask'].cuda(), \
                        batch['label_proportion'].cuda()

                    img = torch.concat([MR, ProstateSurface, target], dim=1)


                    biopsy_all = model(img, grid_sample).float()

                    outputs = torch.masked_select(biopsy_all, mask)
                    val_label = torch.masked_select(label_Gleason, mask).float()

                    target_outputs = torch.masked_select(biopsy_all, target_mask)
                    target_val_label = torch.masked_select(label_Gleason, target_mask).float()

                    system_outputs = torch.masked_select(biopsy_all, system_mask)
                    system_val_label = torch.masked_select(label_Gleason, system_mask).float()


                    y_pred = torch.cat([y_pred, outputs], dim=0)
                    # val_label = torch.stack([post_label(i) for i in val_label], dim=0)
                    y = torch.cat([y, val_label], dim=0)

                    y_pred_tar = torch.cat([y_pred_tar, target_outputs], dim=0)
                    # val_label = torch.stack([post_label(i) for i in val_label], dim=0)
                    y_tar = torch.cat([y_tar, target_val_label], dim=0)

                    y_pred_sys = torch.cat([y_pred_sys, system_outputs], dim=0)
                    # val_label = torch.stack([post_label(i) for i in val_label], dim=0)
                    y_sys = torch.cat([y_sys, system_val_label], dim=0)



                acc_value_0 = torch.eq(post_pred(y_pred), y)
                acc_metric_0 = acc_value_0.sum().item() / len(acc_value_0)

                acc_value_tar = torch.eq(post_pred(y_pred_tar), y_tar)
                acc_metric_tar = acc_value_tar.sum().item() / len(acc_value_tar)

                acc_value_sys = torch.eq(post_pred(y_pred_sys), y_sys)
                acc_metric_sys = acc_value_sys.sum().item() / len(acc_value_sys)



                y_pred_act_0 = post_pre_pred( y_pred)
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



                if best_metric  < auc_result_sys:
                    best_metric = auc_result_sys
                    best_metric_epoch = epoch + 1

                print(args.log_dir)
                print(
                    "  current epoch: {}  current accuracy0  vanilla: {:.4f} current AUC: {:.4f}   best_metric: {:.4f} at epoch {} ".format(
                         epoch + 1, acc_metric_0,
                        auc_result_0, best_metric, best_metric_epoch
                    )
                )
                print(
                    "  current epoch: {}  current accuracy_tar vanilla: {:.4f} current AUC: {:.4f}   ".format(
                         epoch + 1, acc_metric_tar,
                        auc_result_tar
                    )
                )
                print(
                    "  current epoch: {}  current accuracy_sys  vanilla: {:.4f} current AUC: {:.4f}   ".format(
                         epoch + 1, acc_metric_sys,
                        auc_result_sys
                    )
                )

                with open(train_log_dir, "a") as f:
                    f.write(
                        "current epoch: {}  current accuracy0  vanilla: {:.4f} current AUC: {:.4f}   best_metric: {:.4f} at epoch {} \n".format(
                             epoch + 1, acc_metric_0,
                            auc_result_0, best_metric, best_metric_epoch
                        )
                    )

                    f.write(
                        "current epoch: {}  current accuracy_tar vanilla: {:.4f} current AUC: {:.4f}   ".format(
                            epoch + 1, acc_metric_tar,
                            auc_result_tar
                        )
                    )
                    f.write('\n')
                    f.write(
                        "current epoch: {}  current accuracy_sys  vanilla: {:.4f} current AUC: {:.4f}   ".format(
                            epoch + 1, acc_metric_sys,
                            auc_result_sys
                        )
                    )
                    f.write('\n')
                    f.close()


                if epoch>int(args.max_epoches*args.early_begin):
                    early_stopping(auc_result_sys, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                # writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    # writer.close()

    with open(train_log_dir, "a") as f:
        f.write(
            f" train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch} \n"

        )
        f.write('\n')
        f.close()



    print('test begin...........................................')

    model.load_state_dict(torch.load(ori_out_dir+"/best_metric_model.pth"))


    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32).cuda()
        y = torch.tensor([], dtype=torch.float32).cuda()

        y_pred_sys = torch.tensor([], dtype=torch.float32).cuda()
        y_sys = torch.tensor([], dtype=torch.float32).cuda()

        y_pred_tar = torch.tensor([], dtype=torch.float32).cuda()
        y_tar = torch.tensor([], dtype=torch.float32).cuda()

        for batch in test_loader:
            # val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
            MR, ProstateSurface, target, grid_sample, mask, label_Gleason, target_mask, system_mask ,proportion = \
                batch["MR_location"].cuda(), \
                batch["ProstateSurface"].cuda(), \
                batch['target'].cuda(), \
                batch['grid_sample'].cuda(), \
                batch['mask'].cuda(), \
                batch['label_Gleason'].cuda(), \
                batch['target_mask'].cuda(), \
                batch['system_mask'].cuda(), \
                batch['label_proportion'].cuda()

            img = torch.concat([MR, ProstateSurface, target], dim=1)

            # outputs = model(img, PSA, grid_sample, mask)
            # biopsy_all = model(img, PSA, grid_sample, mask).float()
            biopsy_all = model(img, grid_sample).float()


            outputs = torch.masked_select(biopsy_all, mask)
            val_label = torch.masked_select(label_Gleason, mask).float()

            target_outputs = torch.masked_select(biopsy_all, target_mask)
            target_val_label = torch.masked_select(label_Gleason, target_mask).float()

            system_outputs = torch.masked_select(biopsy_all, system_mask)
            system_val_label = torch.masked_select(label_Gleason, system_mask).float()

            y_pred = torch.cat([y_pred, outputs], dim=0)
            # val_label = torch.stack([post_label(i) for i in val_label], dim=0)
            y = torch.cat([y, val_label], dim=0)

            y_pred_tar = torch.cat([y_pred_tar, target_outputs], dim=0)
            # val_label = torch.stack([post_label(i) for i in val_label], dim=0)
            y_tar = torch.cat([y_tar, target_val_label], dim=0)

            y_pred_sys = torch.cat([y_pred_sys, system_outputs], dim=0)
            # val_label = torch.stack([post_label(i) for i in val_label], dim=0)
            y_sys = torch.cat([y_sys, system_val_label], dim=0)

        acc_value_0 = torch.eq(post_pred(y_pred), y)
        acc_metric_0 = acc_value_0.sum().item() / len(acc_value_0)

        acc_value_tar = torch.eq(post_pred(y_pred_tar), y_tar)
        acc_metric_tar = acc_value_tar.sum().item() / len(acc_value_tar)

        acc_value_sys = torch.eq(post_pred(y_pred_sys), y_sys)
        acc_metric_sys = acc_value_sys.sum().item() / len(acc_value_sys)

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

    print(args.log_dir)
    print(
        "  current epoch: {}  current accuracy0  vanilla: {:.4f} current AUC: {:.4f}   best_metric: {:.4f} at epoch {} ".format(
            epoch + 1, acc_metric_0,
            auc_result_0, best_metric, best_metric_epoch
        )
    )
    print(
        "  current epoch: {}  current accuracy_tar vanilla: {:.4f} current AUC: {:.4f}   ".format(
            epoch + 1, acc_metric_tar,
            auc_result_tar
        )
    )
    print(
        "  current epoch: {}  current accuracy_sys  vanilla: {:.4f} current AUC: {:.4f}   ".format(
            epoch + 1, acc_metric_sys,
            auc_result_sys
        )
    )

    with open(train_log_dir, "a") as f:
        f.write(
            "fold{}  current epoch: {}  current accuracy0  vanilla: {:.4f} current AUC: {:.4f}   best_metric: {:.4f} at epoch {} \n".format(
                k_fold, epoch + 1, acc_metric_0,
                auc_result_0, best_metric, best_metric_epoch
            )
        )
        f.write('\n')
        f.write(
            "  current epoch: {}  current accuracy_tar vanilla: {:.4f} current AUC: {:.4f}   ".format(
                epoch + 1, acc_metric_tar,
                auc_result_tar
            )
        )
        f.write('\n')
        f.write(
            "  current epoch: {}  current accuracy_sys  vanilla: {:.4f} current AUC: {:.4f}   ".format(
                epoch + 1, acc_metric_sys,
                auc_result_sys
            )
        )
        f.write('\n')
        f.close()

    acc_y=post_pred(y_pred)
    acc_y_tar = post_pred(y_pred_tar)
    acc_y_sys = post_pred(y_pred_sys)


    F1_score,ACC,SEN,SPE,roc_auc=get_result(y.cpu().numpy(), acc_y.cpu().numpy(), y_pred_act_0.cpu().numpy())
    np.save(out_dir+'/result_save/y.npy', y.cpu().numpy())
    np.save(out_dir + '/result_save/acc_y.npy', acc_y.cpu().numpy())
    np.save(out_dir + '/result_save/y_pred_act.npy', y_pred_act_0.cpu().numpy())

    F1_score_tar, ACC_tar, SEN_tar, SPE_tar, roc_auc_tar = get_result(y_tar.cpu().numpy(), acc_y_tar.cpu().numpy(), y_pred_act_tar.cpu().numpy())
    np.save(out_dir + '/result_save/y_tar.npy', y.cpu().numpy())
    np.save(out_dir + '/result_save/acc_y_tar.npy', acc_y.cpu().numpy())
    np.save(out_dir + '/result_save/y_pred_act_tar.npy', y_pred_act_0.cpu().numpy())

    F1_score_sys, ACC_sys, SEN_sys, SPE_sys, roc_auc_sys = get_result(y_sys.cpu().numpy(), acc_y_sys.cpu().numpy(), y_pred_act_sys.cpu().numpy())
    np.save(out_dir + '/result_save/y_sys.npy', y_sys.cpu().numpy())
    np.save(out_dir + '/result_save/acc_y_sys.npy', acc_y_sys.cpu().numpy())
    np.save(out_dir + '/result_save/y_pred_act_sys.npy', y_pred_act_sys.cpu().numpy())



    with open(train_log_dir, "a") as f:
        f.write('\n' *3)
        f.write('\n')
        f.write('F1_score:' + str(F1_score) + '\n')
        f.write('ACC:' + str(ACC) + '\n')
        f.write('SEN:' + str(SEN) + '\n')
        f.write('SPE:' + str(SPE) + '\n')
        f.write('roc_auc:' + str(roc_auc) + '\n')
        f.write('\n')
        f.write('F1_score_tar:' + str(F1_score_tar) + '\n')
        f.write('ACC_tar:' + str(ACC_tar) + '\n')
        f.write('SEN_tar:' + str(SEN_tar) + '\n')
        f.write('SPE_tar:' + str(SPE_tar) + '\n')
        f.write('roc_auc_tar:' + str(roc_auc_tar) + '\n')
        f.write('\n')
        f.write('F1_score_sys:' + str(F1_score_sys) + '\n')
        f.write('ACC_sys:' + str(ACC_sys) + '\n')
        f.write('SEN_sys:' + str(SEN_sys) + '\n')
        f.write('SPE_sys:' + str(SPE_sys) + '\n')
        f.write('roc_auc_sys:' + str(roc_auc_sys) + '\n')

        f.close()





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parser.add_argument("--data_dir", default='./dataprocessed/full_all_files_name.npy', type=str)

    parser.add_argument("--data_train", default='data_record/divided_data_0mm_7_30points_proportion_mask_SEED0/train.json', type=str)
    parser.add_argument("--data_test", default='data_record/divided_data_0mm_7_30points_proportion_mask_SEED0/test.json', type=str)

    parser.add_argument("--log_dir", default='./record/unet_train_30points_proportion_SEED0_max/', type=str)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--max_epoches", default=100, type=int)


    parser.add_argument("--train_choice", default=0.4, type=float)
    parser.add_argument("--replace_rate", default=0.2, type=float)

    parser.add_argument("--patience", default=7, type=int)
    parser.add_argument("--early_begin", default=0.4, type=float)
    parser.add_argument("--pos_weight", default=17, type=float)

    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--val_interval", default=1, type=int)
    parser.add_argument("--val_seed", default=6, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--gpu_id", default='2', type=str)

    args = parser.parse_args()
    print(args)

    main(args)






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
