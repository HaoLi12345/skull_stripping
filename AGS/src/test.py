import os
from AGS.src.Utils import Utils
import torch
import torch.nn as nn
import numpy as np
from AGS.src.model import subcortical_seg_model, vanila_3dunet
from AGS.src.model.subcortical_seg_model import SingleConv, DoubleConv, ExtResNetBlock
import torch.optim as optim
from AGS.src.dataloader import dataloader
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import cc3d
import pandas

def creater_outputs_folders(prefix=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, 'outputFiles')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if prefix:
        network = 'Networks_' + str(prefix)
        prediction = 'Prediction_' + str(prefix)
    else:
        network = 'Networks'
        prediction = 'Prediction'
    model_dir = os.path.join(output_dir, network)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    prediction_dir = os.path.join(output_dir, prediction)
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    testing_dir = os.path.join(prediction_dir, 'testing')
    if not os.path.exists(testing_dir):
        os.makedirs(testing_dir)
    validation_dir = os.path.join(prediction_dir, 'validation')
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    return output_dir, model_dir, prediction_dir, testing_dir, validation_dir


def initialization(test_image_dir, test_label_dir, test_roi_dir, model_dir, epoch_index, nClass, basic_module_encoder=DoubleConv, basic_module_decoder=ExtResNetBlock,
                   canny=False, vanila=False):
    if canny:
        model = subcortical_seg_model.Unet3d(2, nClass, layer_order='cbr', basic_module_encoder=basic_module_encoder, basic_module_decoder=basic_module_decoder, num_levels=5, f_maps=[16, 32, 64, 64, 128])
    else:
        if vanila:
            model = vanila_3dunet.Unet3d_vanila(1, nClass, f_maps=16)
        else:
            model = subcortical_seg_model.Unet3d(1, nClass, layer_order='cbr', basic_module_encoder=basic_module_encoder, basic_module_decoder=basic_module_decoder, num_levels=5,
                                             f_maps=16)
    model.cuda()
    #print(model)

    epoch_name = sorted(os.listdir(model_dir))[epoch_index]
    path = os.path.join(model_dir, epoch_name)

    loss_criterion = nn.CrossEntropyLoss()
    loss_criterion = nn.MSELoss()

#    model.load_state_dict(torch.load(path))

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    dataset_test = dataloader.SubcorticalDataset(test_image_dir, test_label_dir, test_roi_dir,
                                                 nClass, mode='validation', skull_stripe=False, canny=canny)
    test_loader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=4)

    return model, loss_criterion, optimizer, test_loader


def get_cc_number(label):
    labels_out = cc3d.connected_components(label)
    N_label = np.max(labels_out)
    return N_label


def check_cc(cc_list, nClass):
    error_list = []
    for i in range(0, len(cc_list)):
        if cc_list[i] != nClass - 1:
            print('index:' + str(i) + 'has more/less than ' + str(nClass-1) + ' connected components, found ' + str(cc_list[i]))
            error_list.append(i)
    return error_list

def largest_cc(label):
    # keep largest cc(connected components) for 8 subcortical structures
    final_label = np.zeros(label.shape)
    for structure_index in range(1, 9):
        label_temp = np.zeros(label.shape, dtype=int)
        label_temp[label == structure_index] = 1
        # keep largest cc
        connectivity = 6  # only 26, 18, and 6 are allowed
        labels_out = cc3d.connected_components(label_temp, connectivity=connectivity)
        N_label = np.max(labels_out)

        volume_list = []
        index_list = []
        for i in range(1, N_label + 1):
            voxel_amount = np.count_nonzero(labels_out == i)
            volume_list.append(voxel_amount)
            index_list.append(i)
        if len(volume_list) > 0:
            volume_list_array = np.asarray(volume_list)
            largest_label_volume = volume_list_array.max()
            largest_label_index = volume_list.index(largest_label_volume)
            label_number_from_cc = index_list[largest_label_index]
            # You can extract individual components like so:
            # labels_output is a binary image with value 0 and label_number_from_cc
            label_output = labels_out * (labels_out == label_number_from_cc)
            label_output[labels_out == label_number_from_cc] = structure_index
            final_label[label_output == structure_index] = structure_index
            # final_label = final_label + label_output
    return final_label


def test(model, test_loader, loss_criterion, testing_saving_dir, nClass, print_dice=True):
    # TODO check validation batch size, make multi-batch size version
    connected_component = []
    model.eval()
    with torch.no_grad():

        test_loss_all_image = 0.0

        for i, data in enumerate(test_loader, 0):
            # labels_output is multi-channel version of label
            # ex: label size in b,x,y,z contains n classes
            # labels_output size in b,n,x,y,z
            inputs, labels_output, labels, labels_dict = data
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
            # labels_output is used for multi-channel losses
            labels_output = labels_output.float().cuda()

            # outputs1, outputs = model(inputs)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            # outputs1 = F.softmax(outputs1, dim=1)
            outputs = F.softmax(outputs, dim=1)
            # loss = loss_criterion(outputs, labels_output) + loss_criterion(outputs1, labels_output)
            # loss = loss_criterion(outputs, labels_output)
            #
            # test_loss_all_image += loss.item()

            numpy_predicted = predicted.cpu().numpy()
            numpy_predicted = np.squeeze(numpy_predicted, axis=0)
            numpy_label = labels.cpu().numpy()
            numpy_label = np.squeeze(numpy_label, axis=0)

            label_name = labels_dict['label_name'][0]
            label_dir = labels_dict['label_dir'][0]
            image_dir = labels_dict['image_dir'][0]
            image_name = labels_dict['image_name'][0]
            _, label_data = Utils.load_nii(label_dir, label_name)
            _, image_data = Utils.load_nii(image_dir, image_name)

            #numpy_predicted = largest_cc(numpy_predicted)
            Utils.save_nii(numpy_predicted, image_data, testing_saving_dir, 'Segmentation_' + image_name)
            connected_component.append(get_cc_number(numpy_predicted))
            if print_dice:
                dice_array = Utils.computeDice(numpy_predicted, numpy_label)
                for d_i in range(len(dice_array)):
                    print(" -------------- DSC (Class {}) : {}".format(str(d_i + 1), dice_array[d_i]))
                print('')

    error_cc_index = check_cc(connected_component, nClass)
    validation_loss = (test_loss_all_image / len(labels_dict['label_num_total']))
    # print(validation_loss)
    return validation_loss, connected_component, error_cc_index



def split_diagnosis_cape(seg_list, prefix=False, mode='diagnosed', date_on_file=True):
    df = pandas.read_csv('/home/hao/Hao/Data/PREDICT_DATA/LogismosBSubjects_Info.csv')

    col_scanid = df['scanid']
    col_external_id = df['external_id']

    if mode == 'diagnosed':
        col_score = df['visit_diagnosis']
        diagnosed_list = []
    if mode == 'cape':
        col_score = df['cap_e_group']
        low_list = []
        med_list = []
        high_list = []

    seg_list = sorted(seg_list)

    new_segmentation_list = []
    # diagnosed_list = []
    control_list = []
    for seg_name in seg_list:
        seg_name_split = seg_name.split('.')[0]
        prefix_seg_name = seg_name_split.split('_')
        if prefix:
            external_id = prefix_seg_name[2]
            if date_on_file:
                scan_id = prefix_seg_name[4]
            else:
                scan_id = prefix_seg_name[3]
        else:
            external_id = prefix_seg_name[1]
            if date_on_file:
                scan_id = prefix_seg_name[3]
            else:
                scan_id = prefix_seg_name[2]
        for i in range(len(col_external_id)):
            if col_external_id[i] == external_id and col_scanid[i] == int(scan_id):
                status = col_score[i]
                if mode == 'diagnosed':
                    if status == 'Diagnosed':
                        diagnosed_list.append(seg_name)
                    else:
                        control_list.append(seg_name)
                if mode == 'cape':
                    if status == 'cont':
                        control_list.append(seg_name)
                    if status == 'low':
                        low_list.append(seg_name)
                    if status == 'med':
                        med_list.append(seg_name)
                    if status == 'high':
                        high_list.append(seg_name)
    if mode == 'diagnosed':
        diagnosed_list.sort()
    if mode == 'cape':
        low_list.sort()
        med_list.sort()
        high_list.sort()
    control_list.sort()
    if mode == 'diagnosed':
        for i in range(0, len(diagnosed_list)):
            new_segmentation_list.append(diagnosed_list[i])
        for i in range(0, len(control_list)):
            new_segmentation_list.append(control_list[i])
    if mode == 'cape':
        for i in range(0, len(high_list)):
            new_segmentation_list.append(high_list[i])
        for i in range(0, len(med_list)):
            new_segmentation_list.append(med_list[i])
        for i in range(0, len(low_list)):
            new_segmentation_list.append(low_list[i])
        for i in range(0, len(control_list)):
            new_segmentation_list.append(control_list[i])
    return new_segmentation_list


def make_csv(seg_dir, label_dir, saving_dir, saving_name, mode, prefix=False, HD=True, date_on_file=True):
    seg_list = sorted(os.listdir(seg_dir))
    for i in reversed(range(0, len(seg_list))):
        name = seg_list[i]
        name_split = name.split('_')
        if name_split[2] == 'LD' and HD:
            seg_list.pop(i)
    label_list = sorted(os.listdir(label_dir))

    dice1 = []
    dice2 = []
    dice3 = []
    dice4 = []

    dice5 = []
    dice6 = []
    dice7 = []
    dice8 = []
    total_dice = []

    sd1 = []
    sd2 = []
    sd3 = []
    sd4 = []

    sd5 = []
    sd6 = []
    sd7 = []
    sd8 = []


    hd1 = []
    hd2 = []
    hd3 = []
    hd4 = []

    hd5 = []
    hd6 = []
    hd7 = []
    hd8 = []


    volume1 = []
    volume2 = []
    volume3 = []
    volume4 = []

    volume5 = []
    volume6 = []
    volume7 = []
    volume8 = []
    total_volume = []

    image_name_list = []

    connected_component = []
    cc_1, cc_2, cc_3, cc_4, cc_5, cc_6, cc_7, cc_8 = [], [], [], [], [], [], [], []
    if HD:
        df = pandas.read_csv('/home/hao/Hao/Data/PREDICT_DATA/LogismosBSubjects_Info.csv')
        if mode == 'diagnosed':
            col_status = df['visit_diagnosis']
        if mode == 'cape':
            col_status = df['cap_e_group']
        col_scanid = df['scanid']
        col_external_id = df['external_id']

        status_list = []
        scanid_list = []
        external_id_list = []
        image_type_list = []

        seg_list = split_diagnosis_cape(seg_list, prefix=prefix, mode=mode, date_on_file=date_on_file)

        for seg_name in seg_list:
            # print(seg_name)
            seg_name_split = seg_name.split('.')[0]
            prefix_seg_name = seg_name_split.split('_')
            if prefix:
                external_id_seg = prefix_seg_name[2]
                if date_on_file:
                    scan_id_seg = prefix_seg_name[4]
                    image_type_seg = prefix_seg_name[5]
                else:
                    scan_id_seg = prefix_seg_name[3]
                    image_type_seg = prefix_seg_name[4]
            else:
                external_id_seg = prefix_seg_name[1]
                if date_on_file:
                    scan_id_seg = prefix_seg_name[3]
                    image_type_seg = prefix_seg_name[4]
                else:
                    scan_id_seg = prefix_seg_name[2]
                    image_type_seg = prefix_seg_name[3]

            for label_name in label_list:
                label_name_split = label_name.split('.')[0]
                prefix_label_name = label_name_split.split('_')
                external_id_label = prefix_label_name[1]
                if date_on_file:
                    scan_id_label = prefix_label_name[3]
                else:
                    scan_id_label = prefix_label_name[2]
                if external_id_seg == external_id_label and int(scan_id_seg) == int(scan_id_label):
                    external_id_list.append(external_id_seg)
                    scanid_list.append(scan_id_seg)
                    image_type_list.append(image_type_seg)
                    image_name_list.append(seg_name)
                    for i in range(len(col_external_id)):
                        if col_external_id[i] == external_id_seg and col_scanid[i] == int(scan_id_seg):
                            status_list.append(col_status[i])

                    segmentation, _ = Utils.load_nii(seg_dir, seg_name)
                    segmentation = segmentation.astype(int)

                    label, _ = Utils.load_nii(label_dir, label_name)
                    label = label.astype(int)

                    cc_seg = cc3d.connected_components(segmentation, connectivity=6)
                    N = np.max(cc_seg)
                    connected_component.append(N)

                    ASD_array = Utils.compute_ASD(segmentation, label, spacing_mm=(1.0, 1.0, 1.0), pred_to_gt=True)
                    sd1.append(ASD_array[0])



                    HD_array = Utils.compute_HD(segmentation, label, spacing_mm=(1.0, 1.0, 1.0), percentage=95)
                    hd1.append(HD_array[0])



                    dice_array = Utils.computeDice(segmentation, label)
                    dice1.append(dice_array[0])

    else:
        for seg_name in seg_list:
            seg_name_split = seg_name.split('_')[2] +'_'+ seg_name.split('_')[3] +'_'+ seg_name.split('_')[4]

            for label_name in label_list:
                label_name_split = label_name.split('_')[0] +'_'+ label_name.split('_')[1] +'_'+ label_name.split('_')[2]
                if seg_name_split == label_name_split:
                    image_name_list.append(seg_name)

                    segmentation, _ = Utils.load_nii(seg_dir, seg_name)
                    segmentation = segmentation.astype(int)

                    label, _ = Utils.load_nii(label_dir, label_name)
                    label = label.astype(int)

                    cc_seg = cc3d.connected_components(segmentation, connectivity=6)
                    N = np.max(cc_seg)
                    connected_component.append(N)

                    # ASD_array = Utils.compute_ASD(segmentation, label, spacing_mm=(1.0, 1.0, 1.0), pred_to_gt=True)
                    # sd1.append(ASD_array[0])
                    #
                    #
                    # HD_array = Utils.compute_HD(segmentation, label, spacing_mm=(1.0, 1.0, 1.0), percentage=95)
                    # hd1.append(HD_array[0])


                    # segmentation = largest_cc(segmentation)

                    dice_array = Utils.computeDice(segmentation, label)
                    dice1.append(dice_array[0])






    if HD:
        Results = {'image_name': image_name_list, 'external_id': external_id_list, 'scanid': scanid_list,
                   'status': status_list,
                   'image_type': image_type_list, 'connected_component': connected_component,
                   'dice1': dice1,

                   'asd1': sd1,

                   'hd1': hd1,

                   # 'dice(binary)': total_dice, 'total_volume': total_volume,
                   # 'label_note': label_note,
                   }

        new_df = pandas.DataFrame(Results,
                              columns=['image_name', 'external_id', 'scanid', 'status', 'image_type',
                                       'connected_component',
                                       'dice1',

                                       'asd1',

                                       'hd1',

                                       ])
    else:
        Results = {'image_name': image_name_list,
                   'connected_component': connected_component,
                   'dice1': dice1,

                   # 'asd1': sd1,
                   #
                   # 'hd1': hd1,


                   }

        new_df = pandas.DataFrame(Results,
                                  columns=['image_name',
                                           'connected_component',
                                           'dice1',

                                           # 'asd1',
                                           #
                                           # 'hd1',

                                           ])
    print(new_df)

    # ask how to use variable in df.to_csv
    name = os.path.join(saving_dir, saving_name + '_results.csv')
    new_df.to_csv(name, index=False)


if __name__ == '__main__':



    test_image_dir = ''
    test_label_dir = ''
    test_roi_dir = test_label_dir

    prefix = '123'
    canny = False
    vanila = False

    saving_scv_name = prefix
    nClass = 2
    epoch_index = 0
    saving_scv_dir = './result_csv'
    if not os.path.exists(saving_scv_dir):
        os.makedirs(saving_scv_dir)
    output_dir, model_dir, prediction_dir, testing_saving_dir, validation_saving_dir = creater_outputs_folders(prefix=prefix)
    model, loss_criterion, optimizer, test_loader = initialization(test_image_dir, test_label_dir, test_roi_dir, model_dir,
                                                                   epoch_index, nClass, basic_module_encoder=ExtResNetBlock, basic_module_decoder=ExtResNetBlock, canny=canny, vanila=vanila)
    test_loss, connected_component, error_cc_index = test(model, test_loader, loss_criterion, testing_saving_dir, nClass,
                                                           print_dice=False)

    #testing_saving_dir = '/media/hao/easystore/subcortical_segmentation/2021_summer/Hao/AGS_cycleGAN/src/spie2022/outputFiles_spie2022/Prediction_agsnet_norm_no_aug/testing'
    # make_csv(testing_saving_dir, test_label_dir,
    #          saving_scv_dir, saving_scv_name, mode='cape', prefix=False, HD=False, date_on_file=False)