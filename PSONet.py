# coding=utf-8

import os.path
import random
import time
from matplotlib.colors import ListedColormap
# import torch.optim.lr_scheduler
import numpy as np
import torch.optim as optim
import torch.utils.data as Data
import send2trash
from sampleselect import sampleselect
from lcnn import *
from function_sup import *


class PSONetRunner:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self._set_seed(44)

    # Fix random seed
    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

    def run(self):
        start_time = time.time()
        args = self.args

        # Load data
        I1 = Apply_load_matdata(args.data_path, args.data1_name, norm_flag=False)  # T1 of bi-temporal
        I2 = Apply_load_matdata(args.data_path, args.data2_name, norm_flag=False)  # T2 of bi-temporal
        nr_height, nc_width, ndim = I1.shape
        # For accuracy analysis(some datasets contain background)
        index = Apply_load_matdata(args.data_path, 'index', norm_flag=False, std_flag=False)
        index = index - 1
        index = np.reshape(index, newshape=[index.shape[0], ])
        # Patches
        X = np.concatenate((I1, I2), axis=2)
        X_patches = Apply_patches(X, win_size=args.patch_size).astype(np.float32)
        X_patches = np.transpose(X_patches, (0, 3, 1, 2))
        X_patches = torch.from_numpy(X_patches)

        # Identify and delete the existing PSONet
        if os.path.isfile(f'{args.save_path}/PSONet.pth'):
            print("note:-----PSONet.pth exists and it will be deleted-----")
            send2trash.send2trash(f'{args.save_path}/PSONet.pth')
            print("note:-----PSONet.pth has been moved to Trash-----")
        else:
            print("note:-----PSONet.pth does not exist，program continue-----")
        # Network initialization
        lcnn_model = lcnn(args.in_channels)
        optimizer_ft = optim.Adam(params=lcnn_model.parameters(), lr=args.learn_rate)

        # Initial change detection
        # Initial change detection--Pseudo-change map based on spectral features(DI+FCM)
        DI_K = abs(I1 - I2)
        label_K, _, _ = Apply_fcm(DI_K, n_clusters=3, adjust=True)  # 0-unchanged, 1-unknown, and 2-changed
        # Initial change detection--Pseudo-change map based on deep features(DCVA+FCM)
        deepfea1 = Apply_load_matdata(args.dcva_path, args.dcva1_name, norm_flag=False, std_flag=False)
        deepfea2 = Apply_load_matdata(args.dcva_path, args.dcva2_name, norm_flag=False, std_flag=False)
        DI_D = Apply_cva(deepfea1, deepfea2, no_sum=False)
        label_D, _, _ = Apply_fcm(DI_D, n_clusters=3, adjust=True)
        # Initial change detection--Pseudo-change map based on class signal domain features(Difference of class+FCM)
        _, _, Vector1_affiliation = Apply_fcm(I1, n_clusters=4, adjust=True)
        _, _, Vector2_affiliation = Apply_fcm(I2, n_clusters=4, adjust=True)
        Vector1_affiliation = np.reshape(Vector1_affiliation, newshape=[nr_height, nc_width, -1])
        Vector2_affiliation = np.reshape(Vector2_affiliation, newshape=[nr_height, nc_width, -1])
        DI_C = Apply_cva(Vector1_affiliation, Vector2_affiliation, no_sum=False)
        label_C, _, _ = Apply_fcm(DI_C, n_clusters=3, adjust=True)
        # Release space
        del I1, I2, X, DI_K, deepfea1, deepfea2, DI_D, Vector1_affiliation, Vector2_affiliation, DI_C

        # “Weak-to-strong” change signal image
        WTSCS = label_K + label_D + label_C
        sio.savemat(f"{args.save_path}/WTSCS.mat", {'label': WTSCS})  # Initial “weak-to-strong” change signal image
        BCDM = np.zeros_like(WTSCS)
        WTSCS = WTSCS + 1  # + 1 to avoid 0
        mask = np.ones_like(BCDM)
        pred = np.zeros_like(BCDM, dtype=int)
        nsum_unknown = 100
        PROGRESSIVE = 1

        # Progression
        while nsum_unknown > 0:
            print('note:-----Progression：', PROGRESSIVE, '----')
            # Samples selection
            # index0 is unchanged,index1 is changed,and index2 is unknown
            index0, index1, index2, WTSCS = sampleselect(WTSCS, pred, mask)
            nsum_unknown = np.sum(index2 == 1)
            # Determine whether it is necessary to enter progression for network training
            if nsum_unknown == 0:  # There are no unknown pixels, and the progression ends
                label0 = np.reshape(index0, newshape=[nr_height, nc_width]) + 0
                label1 = np.reshape(index1, newshape=[nr_height, nc_width]) + 0
                BCDM = BCDM + label0 + 2 * label1
            else:  # There are unknown pixels, and the progression continues
                # Mask fusion
                label0 = np.reshape(index0, newshape=[nr_height, nc_width]) + 0
                label1 = np.reshape(index1, newshape=[nr_height, nc_width]) + 0
                BCDM = BCDM + label0 + 2 * label1  # changed is 2 , unchanged is 1, unknown is 0
                # Index extraction for training and test samples
                nsum0 = np.sum(label0 == 1)
                tra_index0 = np.where(index0 == True)[0].tolist()
                nsum1 = np.sum(label1 == 1)
                tra_index1 = np.where(index1 == True)[0].tolist()
                # Index and ground truth of the training samples
                tra_index = tra_index0
                tra_index.extend(tra_index1)
                tra_label = np.zeros(len(tra_index), dtype=int)
                tra_label[-len(tra_index1):] = 1
                # Index of test samples
                test_index = np.where(index2 == True)[0].tolist()
                # Training samples and test samples
                # Training samples
                train_data = X_patches[tra_index, :, :, :]
                tra_label = torch.from_numpy(tra_label).float()
                train_dataset = Data.TensorDataset(train_data, tra_label)
                BATCH_SIZE = int(np.ceil(train_data.shape[0] * 0.04))  # batch size for training
                train_dataset_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, )
                del train_data, tra_label, train_dataset
                # Test samples
                test_data = X_patches[test_index, :, :, :]
                BATCH_SIZE_tes = int(np.ceil(test_data.shape[0] * 0.03))  # batch size for test
                test_dataset_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE_tes, shuffle=False, )
                del test_data

                # Network training
                if args.Train:
                    try:
                        lcnn_model.load_state_dict(torch.load(f'{args.save_path}/PSONet.pth'))
                        print("note:------found existing model-------")
                    except:
                        print("note:------no existing model found-------")
                    lcnn_model.to(self.device)
                    lcnn_model.train()
                    for epoch in range(args.epoch):
                        runing_loss = 0.0
                        for batch_x, batch_y in train_dataset_loader:
                            batch_x = batch_x.to(self.device)
                            batch_y = torch.squeeze(batch_y)
                            batch_y = batch_y.to(self.device)
                            optimizer_ft.zero_grad()
                            with torch.set_grad_enabled(args.Train):
                                out = lcnn_model(batch_x)
                                out = torch.squeeze(out)
                                # Weighted Bceloss
                                weight = torch.zeros_like(out).float()
                                weight = torch.fill_(weight, nsum1 / (nsum0 + nsum1))
                                weight[batch_y > 0] = nsum0 / (nsum0 + nsum1)
                                loss = nn.BCELoss(weight=weight)(out, batch_y)
                                # Backward
                                loss.backward()
                                optimizer_ft.step()
                            runing_loss += loss.item() * batch_x.size(0)  # running loss
                        epoch_loss = runing_loss / len(train_dataset_loader.dataset)  # epoch loss
                        print('epochs:', epoch, 'loss: {:.4f}'.format(epoch_loss))
                    # Save the network trained in current progression for transfer training in following progression
                    torch.save(lcnn_model.state_dict(), f'{args.save_path}/PSONet.pth')
                del train_dataset_loader

                # Network testing
                if args.Test:
                    with torch.no_grad():
                        lcnn_model.load_state_dict(torch.load(f'{args.save_path}/PSONet.pth'))
                        lcnn_model.to(self.device)
                        lcnn_model.eval()
                        out = torch.empty((0, 1))
                        for batch_x in test_dataset_loader:
                            batch_x = batch_x.to(self.device)
                            out_y = lcnn_model(batch_x)
                            out = torch.cat((out, out_y), axis=0)
                    pred_progression = out.detach().cpu().numpy()
                    pred_progression = np.int32(pred_progression > 0.5)
                    pred_progression = np.reshape(pred_progression, newshape=[pred_progression.shape[0], ])
                    pred = np.zeros_like(BCDM, dtype=int)
                    pred = np.reshape(pred, newshape=[nr_height * nc_width, ])
                    pred[test_index] = pred_progression
                    pred = np.reshape(pred, newshape=[nr_height, nc_width])
                del test_dataset_loader
                # Update mask
                mask = np.reshape(index2, newshape=[nr_height, nc_width])
                # Save the network trained in each progression for comparison
            torch.save(lcnn_model.state_dict(), f'{args.save_path}/PSONet_{PROGRESSIVE}.pth')
            PROGRESSIVE += 1

        # Obtain the final binary change detection map
        BCDM = BCDM - 1
        end_time = time.time()
        print('note:-----running time：', end_time - start_time, 'second', '----')

        # Accuracy calculation and saving
        GT = sio.loadmat(os.path.join(args.data_path, args.gt_name))['gt']
        TP, TN, FP, FN, OA, TE, Kappa, F1, Recall, Precision, FA, MA = Apply_accuracy_binary(BCDM, GT, index,fourshow=False)
        # Write and save accuracy
        with open(f'{args.save_path}/ACCURACY.txt', 'w') as f:
            f.write(
                f"TP: {TP}, \nTN: {TN}, \nFP: {FP}, \nFN: {FN}, \nOA: {OA}, \nTE: {TE}, \nKAPPA: {Kappa}, \nF1: {F1}, "
                f"\nRecall: {Recall}, \nPrecision: {Precision}, \nFA: {FA}, \nMA: {MA}, \nTime: {end_time - start_time}")
            f.close()
        # Save the final binary change detection map(.mat format)
        sio.savemat(f"{args.save_path}/BCDM.mat", {'bcdm': BCDM})
        # Save the final binary change detection map(.png format)
        mycolor = [(61 / 255, 38 / 255, 168 / 255),
                   (249 / 255, 250 / 255, 20 / 255)]
        cmap = ListedColormap(mycolor)
        plt.imshow(BCDM, cmap=cmap)
        plt.axis('off')
        plt.savefig(fname=f'{args.save_path}/BCDM.png', dpi=600, bbox_inches='tight',
                    pad_inches=0)
        print('-----------Progression is over--------------')