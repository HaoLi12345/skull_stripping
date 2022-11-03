import os
from os.path import dirname
import nibabel as nib
import numpy as np
import pandas
import shutil
from AGS.src.Utils import Utils
import time
import torch.nn as nn
import math
import gc
"get new folders, order of the output is image,label,roi"
from surface_distance import compute_surface_distances, compute_average_surface_distance, compute_robust_hausdorff


def remove_dir_from_parent_directory(foldername):
    current_directory = os.getcwd()
    parent_directory = dirname(current_directory)
    sub_directory = os.path.join(parent_directory, foldername)
    if os.path.exists(sub_directory):
        os.rmdir(sub_directory)


def create_new_folder_from_parent_directory(foldername, subfoldername, imagefolder, labelfolder, roifolder):
    new_foldlist = []
    folderlist = []
    folderlist.append(imagefolder)
    folderlist.append(labelfolder)
    folderlist.append(roifolder)
    current_directory = os.getcwd()
    parent_directory = dirname(current_directory)

    sub_directory = os.path.join(parent_directory, foldername)
    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)
    final_directory = os.path.join(sub_directory, subfoldername)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for i in range(0, len(folderlist)):
        new_directory = os.path.join(final_directory, folderlist[i])
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        new_foldlist.append(new_directory)
    new_image_dir = new_foldlist[0]
    new_label_dir = new_foldlist[1]
    new_roi_dir = new_foldlist[2]
    print("The current working directory is " + str(current_directory))
    print("The parent working directory is " + str(parent_directory))
    print("The new folder is created " + str(sub_directory))
    print("The new folder is created " + str(final_directory))
    print(' folders are created')

    return new_image_dir, new_label_dir, new_roi_dir


def load_nii(dir, imagename):
    image_dir = os.path.join(dir, imagename)
    image_data = nib.load(image_dir)
    image = image_data.get_fdata()
    return image, image_data


def save_nii(image, image_data, dir, name):

    new_I = nib.Nifti1Image(image, image_data.affine, image_data.header)
    #new_I = nib.Nifti1Image(image, image_data)
    if dir[-1] == '/':
        dir = dir
    else:
        dir = dir + '/'
    nib.save(new_I, dir + name)
    # print(' image: ' + name + ' saved to ' + dir)


def find_roi_border(roi):
    """
    nib.load reads the image as the dimension [X, Y, Z]
    np.where function returns pixel location as the order [X, Y, Z]
    where X is the column of the image and Y is the row of the image
    """

    roi_voxel_location = np.where(roi == 1)

    roi_voxel_z = roi_voxel_location[2]
    roi_voxel_y = roi_voxel_location[1]
    roi_voxel_x = roi_voxel_location[0]

    roi_max_x = np.amax(roi_voxel_x)
    roi_min_x = np.amin(roi_voxel_x)

    roi_max_y = np.amax(roi_voxel_y)
    roi_min_y = np.amin(roi_voxel_y)

    roi_max_z = np.amax(roi_voxel_z)
    roi_min_z = np.amin(roi_voxel_z)

    return roi_min_x, roi_max_x, roi_min_y, roi_max_y, roi_min_z, roi_max_z


def crop_image(roi, roi_min_x, roi_max_x, roi_min_y, roi_max_y, roi_min_z, roi_max_z):
    new_roi = roi[roi_min_x: roi_max_x,
              roi_min_y: roi_max_y,
              roi_min_z: roi_max_z]
    return new_roi


def split_training_validation(image_list, validation_index):
    validation_list = []
    image_list = sorted(image_list)
    new_image_list = image_list
    for i in validation_index:
        validation_list.append(image_list[i])
    for i in sorted(validation_index, reverse=True):
        new_image_list.pop(i)
    return new_image_list, validation_list


def crop_and_patch_image(image_list, image_dir,
                         label_list, label_dir,
                         roi_list, roi_dir,
                         new_folder_dir_c, new_folder_dir_p,
                         sample_size, step_size, status  # 0 training 1 validation 2 testing
                         ):
    t = time.time()
    names_list = []
    x_max_total_coordinate = []
    y_max_total_coordinate = []
    z_max_total_coordinate = []

    x_min_total_coordinate = []
    y_min_total_coordinate = []
    z_min_total_coordinate = []

    x_shape = []
    y_shape = []
    z_shape = []
    for i in range(0, len(image_list)):
        [image, image_data] = Utils.load_nii(image_dir, image_list[i])
        # print(image_list[i])
        # print(image.shape[0])
        [label, label_data] = Utils.load_nii(label_dir, label_list[i])
        [roi, roi_data] = Utils.load_nii(roi_dir, roi_list[i])

        roi.astype(int)
        [roi_min_x, roi_max_x,
         roi_min_y, roi_max_y,
         roi_min_z, roi_max_z] = Utils.find_roi_border(roi)
        """
        remember, python doesn't include the last element if we do a:b
        a:b equals [a,b) or range(a,b)
        """
        new_image = Utils.crop_image(image, roi_min_x, roi_max_x + 1, roi_min_y, roi_max_y + 1, roi_min_z,
                                     roi_max_z + 1)
        new_label = Utils.crop_image(label, roi_min_x, roi_max_x + 1, roi_min_y, roi_max_y + 1, roi_min_z,
                                     roi_max_z + 1)
        new_roi = Utils.crop_image(roi, roi_min_x, roi_max_x + 1, roi_min_y, roi_max_y + 1, roi_min_z, roi_max_z + 1)

        name_image = image_list[i]
        if name_image.endswith(".nii.gz"):
            name_image = name_image[0:-7]

        name_label = label_list[i]
        if name_label.endswith(".nii.gz"):
            name_label = name_label[0:-7]

        name_roi = roi_list[i]
        if name_roi.endswith(".nii.gz"):
            name_roi = name_roi[0:-7]

        Utils.save_nii(new_image, image_data, new_folder_dir_c[0], name_image + '_crop.nii.gz')
        Utils.save_nii(new_label, label_data, new_folder_dir_c[1], name_label + '_crop.nii.gz')
        Utils.save_nii(new_roi, roi_data, new_folder_dir_c[2], name_roi + '_crop.nii.gz')

        """
        we want to include all boundary or border point, if we dont +1, we will lose one point
        x_distance means the totally pixels alone the x direction
        """
        x_distance = roi_max_x + 1 - roi_min_x
        y_distance = roi_max_y + 1 - roi_min_y
        z_distance = roi_max_z + 1 - roi_min_z

        ratio = int(sample_size / step_size)
        """
        since there is a round function, the samples are not exactly accurate.
        """
        # x_number = int(round(x_distance / sample_size)) * sample_size / step_size

        x_number = int(round(x_distance / step_size))
        y_number = int(round(y_distance / step_size))
        z_number = int(round(z_distance / step_size))
        """
        after testing, we only need those center areas(9*9) to put back to the image
        """
        start_point = [roi_min_x, roi_min_y, roi_min_z]
        x_coordinate = []
        y_coordinate = []
        z_coordinate = []

        for sample_num_x in range(0, x_number + 1):
            x_coordinate.append(start_point[0] + sample_num_x * step_size)
        for sample_num_y in range(0, y_number + 1):
            y_coordinate.append(start_point[1] + sample_num_y * step_size)
        for sample_num_z in range(0, z_number + 1):
            z_coordinate.append(start_point[2] + sample_num_z * step_size)
        index = 1
        a = []
        for ii in range(0, z_number - ratio + 1):
            a.append(z_coordinate[ii])
        for i_x in range(0, x_number - ratio + 1):
            crop_xcoord_min = x_coordinate[i_x]
            crop_xcoord_max = x_coordinate[i_x + ratio]
            for i_y in range(0, y_number - ratio + 1):
                crop_ycoord_min = y_coordinate[i_y]
                crop_ycoord_max = y_coordinate[i_y + ratio]
                for i_z in range(0, z_number - ratio + 1):
                    crop_zcoord_min = z_coordinate[i_z]
                    crop_zcoord_max = z_coordinate[i_z + ratio]

                    new_patched_image = Utils.crop_image(image, crop_xcoord_min, crop_xcoord_max
                                                         , crop_ycoord_min, crop_ycoord_max
                                                         , crop_zcoord_min, crop_zcoord_max)
                    new_patched_label = Utils.crop_image(label, crop_xcoord_min, crop_xcoord_max
                                                         , crop_ycoord_min, crop_ycoord_max
                                                         , crop_zcoord_min, crop_zcoord_max)
                    new_patched_roi = Utils.crop_image(roi, crop_xcoord_min, crop_xcoord_max
                                                       , crop_ycoord_min, crop_ycoord_max
                                                       , crop_zcoord_min, crop_zcoord_max)
                    # print(str(index))
                    # print(image[crop_xcoord_min, crop_ycoord_min, crop_zcoord_min])
                    # print(new_patched_image[0, 0, 0])
                    # print(image[crop_xcoord_max - 1, crop_ycoord_max - 1, crop_zcoord_max - 1])
                    # print(new_patched_image[-1, -1, -1])
                    # print('_____________________________________________')
                    Utils.save_nii(new_patched_image, image_data, new_folder_dir_p[0],
                                   name_image + str(index).zfill(5) + '_pathched_' + '.nii.gz')
                    Utils.save_nii(new_patched_label, label_data, new_folder_dir_p[1],
                                   name_label + str(index).zfill(5) + '_pathched_' + '.nii.gz')
                    Utils.save_nii(new_patched_roi, roi_data, new_folder_dir_p[2],
                                   name_roi + str(index).zfill(5) + '_pathched_' + '.nii.gz')

                    """
                    for making csv file
                    """
                    names_list.append(name_image + str(index).zfill(5) + '_pathched_' + '.nii.gz')
                    x_max_total_coordinate.append(crop_xcoord_max)
                    y_max_total_coordinate.append(crop_ycoord_max)
                    z_max_total_coordinate.append(crop_zcoord_max)

                    x_min_total_coordinate.append(crop_xcoord_min)
                    y_min_total_coordinate.append(crop_ycoord_min)
                    z_min_total_coordinate.append(crop_zcoord_min)

                    x_shape.append(image.shape[0])
                    y_shape.append(image.shape[1])
                    z_shape.append(image.shape[2])

                    index = index + 1
        print('image : ' + str(image_list[i]) + ' done ')
    elapse = time.time() - t
    print(' total time spent :' + str(elapse))

    Results = {'image_name': names_list, 'x_min': x_min_total_coordinate, 'x_max': x_max_total_coordinate,
               'y_min': y_min_total_coordinate, 'y_max': y_max_total_coordinate,
               'z_min': z_min_total_coordinate, 'z_max': z_max_total_coordinate,
               'x_shape': x_shape, 'y_shape': y_shape, 'z_shape': z_shape

               }

    df = pandas.DataFrame(Results,
                          columns=['image_name', 'x_min', 'x_max',
                                   'y_min', 'y_max',
                                   'z_min', 'z_max',
                                   'x_shape', 'y_shape', 'z_shape'
                                   ])
    # print(df)
    df.to_csv('coordinates_table.csv', index=False)
    if status == 0:
        shutil.move(os.getcwd() + '/coordinates_table.csv',
                    dirname(dirname(new_folder_dir_p[0])) + '/coordinates_table_training.csv')
    if status == 1:
        shutil.move(os.getcwd() + '/coordinates_table.csv',
                    dirname(dirname(new_folder_dir_p[0])) + '/coordinates_table_validation.csv')
    if status == 2:
        shutil.move(os.getcwd() + '/coordinates_table.csv',
                    dirname(dirname(new_folder_dir_p[0])) + '/coordinates_table_testing.csv')



def compute_ASD(seg, gt, spacing_mm=(1.0, 1.0, 1.0), pred_to_gt=True):
    n_classes = int(np.max(gt) + 1)

    ASDarray = []

    for c_i in range(1, n_classes):

        mask_pred = np.zeros(seg.shape, dtype=np.bool)
        mask_pred[seg == c_i] = 1


        mask_gt = np.zeros(gt.shape, dtype=np.bool)
        mask_gt[gt == c_i] = 1

        surface_distances = compute_surface_distances(
            mask_gt, mask_pred, spacing_mm=spacing_mm)
        avg_surf_dist = compute_average_surface_distance(surface_distances)
        if pred_to_gt:
            ASDarray.append(avg_surf_dist[1])
        else:
            ASDarray.append(avg_surf_dist[0])

    return ASDarray


def compute_HD(seg, gt, spacing_mm=(1.0, 1.0, 1.0), percentage=100):
    n_classes = int(np.max(gt) + 1)

    HDarray = []

    for c_i in range(1, n_classes):

        mask_pred = np.zeros(seg.shape, dtype=np.bool)
        mask_pred[seg == c_i] = 1


        mask_gt = np.zeros(gt.shape, dtype=np.bool)
        mask_gt[gt == c_i] = 1

        surface_distances = compute_surface_distances(
            mask_gt, mask_pred, spacing_mm=spacing_mm)
        hd_surf_dist = compute_robust_hausdorff(surface_distances, percentage)

        HDarray.append(hd_surf_dist)


    return HDarray

def computeDice(autoSeg, groundTruth):
    """ Returns
    -------
    DiceArray : floats array

          Dice coefficient as a float on range [0,1].
          Maximum similarity = 1
          No similarity = 0 """

    n_classes = int(np.max(groundTruth) + 1)

    DiceArray = []

    for c_i in range(1, n_classes):
        idx_Auto = np.where(autoSeg.flatten() == c_i)[0]
        idx_GT = np.where(groundTruth.flatten() == c_i)[0]

        autoArray = np.zeros(autoSeg.size, dtype=np.bool)
        autoArray[idx_Auto] = 1

        gtArray = np.zeros(autoSeg.size, dtype=np.bool)
        gtArray[idx_GT] = 1

        dsc = dice(autoArray, gtArray)

        # dice = np.sum(autoSeg[groundTruth==c_i])*2.0 / (np.sum(autoSeg) + np.sum(groundTruth))
        DiceArray.append(dsc)

    return DiceArray


def dice(im1, im2):
    """
    Computes the Dice coefficient
    ----------
    im1 : boolean array
    im2 : boolean array

    If they are not boolean, they will be converted.

    -------
    It returns the Dice coefficient as a float on the range [0,1].
        1: Perfect overlapping
        0: Not overlapping
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.size != im2.size:
        raise ValueError("Size mismatch between input arrays!!!")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 1.0

    # Compute Dice
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
                y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc


def random_sampler(image_dir, label_dir, sample_size, p_foreground,
                   total_sample_num, validation_index, use_roi,
                   padding_bool, receptive_field, skull_strip, use_xyz, elastic, affine, intensity):

    tic = time.time()
    index = 0

    image_list = sorted(os.listdir(image_dir))
    label_list = sorted(os.listdir(label_dir))


    validation_index = sorted(validation_index, reverse=True)
    if len(validation_index) != 0:
        for i_vali in validation_index:
            image_list.pop(i_vali)
            label_list.pop(i_vali)


    if len(image_list) != len(label_list):
        raise ValueError(' number of images are not same as number of labels')


    total_image_sample_list = []
    total_label_sample_list = []


    """
    we assume our sample is a cube, if not, change following code
    """
    if sample_size % 2 == 0:
        half_sample = sample_size / 2
        half_sample_size_to_put = [half_sample - 1, half_sample]
    else:
        half_sample = math.floor(sample_size / 2)
        half_sample_size_to_put = [half_sample, half_sample]

    p_background = 1 - p_foreground

    total_image_num = len(image_list)
    sample_num_each_image = round(total_sample_num / total_image_num)
    sample_num_foreground = round(sample_num_each_image * p_foreground)
    sample_num_background = sample_num_each_image - sample_num_foreground
    sample_num_fore_back = [sample_num_foreground, sample_num_background]

    for i in range(0, len(image_list)):
        image, image_data = Utils.load_nii(image_dir, image_list[i])
        label, label_data = Utils.load_nii(label_dir, label_list[i])


        if padding_bool == 1:
            # dont need padding value for training sampling
            image, _ = apply_padding(image, receptive_field)
            label, _ = apply_padding(label, receptive_field)

        if use_roi == 0:
            roi = np.ones([image.shape[0], image.shape[1], image.shape[2]], dtype=np.int)
        if skull_strip == 1:
            image = np.multiply(image, roi)


        if image.shape != label.shape:
            raise ValueError(' shape of images are not same as shape of labels')
        if image.shape != label.shape:
            raise ValueError(' shape of images are not same as shape of rois')
        if roi.shape != label.shape:
            raise ValueError(' shape of rois are not same as shape of labels')
        """
        set the border where we take central coordinate of samples
        we assume our sample is a cube, if not, change following code
        """

        tic1 = time.time()
        border_x_min = half_sample_size_to_put[0]
        border_x_max = image.shape[0] - half_sample_size_to_put[1]

        border_y_min = half_sample_size_to_put[0]
        border_y_max = image.shape[1] - half_sample_size_to_put[1]

        border_z_min = half_sample_size_to_put[0]
        border_z_max = image.shape[2] - half_sample_size_to_put[1]

        mask = np.zeros(image.shape, dtype='int32')
        mask[border_x_min:border_x_max, border_y_min:border_y_max, border_z_min:border_z_max] = 1

        foreground_mask = (label > 0).astype(int)
        background_mask = (roi > 0) * (foreground_mask == 0)
        """
        this is only for some uncommon cases
        manually set the border is 0 no matter the useful or interesting part or label in that area
        """
        list_fore_back_ground = [foreground_mask, background_mask]
        #tic1 = time.time()

        for j in range(0, 2):
            mask_coor = list_fore_back_ground[0] * mask

            p_mask = mask_coor / (1.0 * np.sum(mask_coor))
            # we need set this p to get rid of where the pixel values are 0
            p_mask_flatten = p_mask.flatten()

            central_index = np.random.choice(mask_coor.size,
                                             size=sample_num_fore_back[j],
                                             replace=True,
                                             p=p_mask_flatten)

            # we want to set is as ndarray not tuple
            central_index_to_coor = np.asarray(np.unravel_index(central_index, mask_coor.shape))

            for k in range(0, central_index_to_coor.shape[1]):
                min_x = central_index_to_coor[0][k] - half_sample_size_to_put[0]
                max_x = central_index_to_coor[0][k] + half_sample_size_to_put[1]

                min_y = central_index_to_coor[1][k] - half_sample_size_to_put[0]
                max_y = central_index_to_coor[1][k] + half_sample_size_to_put[1]

                min_z = central_index_to_coor[2][k] - half_sample_size_to_put[0]
                max_z = central_index_to_coor[2][k] + half_sample_size_to_put[1]
                if use_xyz == 1:
                    image_sample_data = image[min_x:max_x + 1, min_y: max_y + 1, min_z:max_z + 1]
                    label_sample_data = label[min_x:max_x + 1, min_y: max_y + 1, min_z:max_z + 1]
                    # get coordinates
                    image_sample_x = np.asarray(list(range(min_x, max_x+1)))
                    image_sample_y = np.asarray(list(range(min_y, max_y+1)))
                    image_sample_z = np.asarray(list(range(min_z, max_z+1)))
                    # change shape, so easy to append rows or columns
                    image_sample_x = np.reshape(image_sample_x, len(range(min_x, max_x+1)))
                    image_sample_y = np.reshape(image_sample_y, len(range(min_y, max_y + 1)))
                    image_sample_z = np.reshape(image_sample_z, len(range(min_z, max_z + 1)))
                    x_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    y_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    z_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    for i_x in range(0, len(range(min_x, max_x+1))):
                        x_coor[i_x, :, :] = image_sample_x[i_x]
                    for i_y in range(0, len(range(min_y, max_y+1))):
                        y_coor[:, i_y, :] = image_sample_y[i_y]
                    for i_z in range(0, len(range(min_z, max_z+1))):
                        z_coor[:, :, i_z] = image_sample_z[i_z]
                    image_sample_data = np.expand_dims(image_sample_data, axis=0)
                    label_sample_data = np.expand_dims(label_sample_data, axis=0)
                    x_coor = np.expand_dims(x_coor, axis=0)
                    y_coor = np.expand_dims(y_coor, axis=0)
                    z_coor = np.expand_dims(z_coor, axis=0)
                    image_sample = np.concatenate((image_sample_data, x_coor, y_coor, z_coor), axis=0)
                    #label_sample = np.concatenate((label_sample_data, x_coor, y_coor, z_coor), axis=0)
                    label_sample = label[min_x:max_x + 1, min_y: max_y + 1, min_z:max_z + 1]

                else:
                    image_sample = image[min_x:max_x + 1, min_y: max_y + 1, min_z:max_z + 1]
                    label_sample = label[min_x:max_x + 1, min_y: max_y + 1, min_z:max_z + 1]


                total_image_sample_list.append(image_sample)
                total_label_sample_list.append(label_sample)


                #gc.collect()
        index = index + 1
        # time0 = time.time()
        # print(time0 - tic1)
    toc = time.time() - tic
    print('image sampling {}/{} done, used {}'.format(index, len(image_list), toc))
    return total_image_sample_list, total_label_sample_list



def apply_padding(image, receptive_field):
    left_padding = int( (receptive_field - 1) / 2)
    right_padding = int(receptive_field - 1 - left_padding)
    # output_image = np.pad(image, ((left_padding, right_padding),
    #                               (left_padding, right_padding),
    #                               (left_padding, right_padding)), 'reflect')
    output_image = np.pad(image, (left_padding, right_padding), 'reflect')
    padding_value = [left_padding, right_padding]
    return output_image, padding_value

def apply_unpadding(image, padding_value):
    left_unpadding = padding_value[0]
    right_unpadding = padding_value[1]
    output_image = image[left_unpadding:-right_unpadding, left_unpadding:-right_unpadding, left_unpadding:-right_unpadding]
    return output_image



def sample_image(image, sample_size, step_size, roi, use_roi, skull_strip, use_xyz):
    if use_roi == 0:
        roi = np.ones([image.shape[0], image.shape[1], image.shape[2]], dtype=int)
    if skull_strip == 1:
        image = np.multiply(image, roi)

    sample_coords = []
    sample_list = []
    zmin_next = 0
    z_stop = False

    while not z_stop:
        zmax = min(zmin_next + sample_size, image.shape[2])
        zmin = zmax - sample_size
        zmin_next = zmin_next + step_size
        if zmax < image.shape[2]:
            z_stop = False
        else:
            z_stop = True

        ymin_next = 0
        y_stop = False
        while not y_stop:
            ymax = min(ymin_next + sample_size, image.shape[1])
            ymin = ymax - sample_size
            ymin_next = ymin_next + step_size
            if ymax < image.shape[1]:
                y_stop = False
            else:
                y_stop = True

            xmin_next = 0

            # print(xmin_next)
            x_stop = False
            while not x_stop:
                # print(xmin_next)
                xmax = min(xmin_next + sample_size, image.shape[0])
                xmin = xmax - sample_size
                xmin_next = xmin_next + step_size
                if xmax < image.shape[0]:
                    x_stop = False
                else:
                    x_stop = True

                if isinstance(roi, np.ndarray):
                    if not np.any(roi[xmin:xmax, ymin:ymax, zmin:zmax]):
                        continue

                sample_coords.append([xmin, ymin, zmin])
                if use_xyz == 1:
                    image_sample_data = image[xmin:xmax, ymin: ymax, zmin:zmax]
                    # get coordinates
                    image_sample_x = np.asarray(list(range(xmin, xmax)))
                    image_sample_y = np.asarray(list(range(ymin, ymax)))
                    image_sample_z = np.asarray(list(range(zmin, zmax)))
                    # change shape, so easy to append rows or columns
                    image_sample_x = np.reshape(image_sample_x, len(list(range(xmin, xmax))))
                    image_sample_y = np.reshape(image_sample_y, len(list(range(ymin, ymax))))
                    image_sample_z = np.reshape(image_sample_z, len(list(range(zmin, zmax))))
                    x_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    y_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    z_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    for i_x in range(0, len(range(xmin, xmax))):
                        x_coor[i_x, :, :] = image_sample_x[i_x]
                    for i_y in range(0, len(range(ymin, ymax))):
                        y_coor[:, i_y, :] = image_sample_y[i_y]
                    for i_z in range(0, len(range(zmin, zmax))):
                        z_coor[:, :, i_z] = image_sample_z[i_z]

                    image_sample_data = np.expand_dims(image_sample_data, axis=0)
                    x_coor = np.expand_dims(x_coor, axis=0)
                    y_coor = np.expand_dims(y_coor, axis=0)
                    z_coor = np.expand_dims(z_coor, axis=0)
                    image_sample = np.concatenate((image_sample_data, x_coor, y_coor, z_coor), axis=0)
                    sample_list.append(image_sample)
                else:
                    sample_list.append(image[xmin:xmax, ymin:ymax, zmin:zmax])


    return sample_coords, sample_list


def sample_image_T1_T2(image_T1, image_T2, sample_size, step_size, roi, use_roi, skull_strip, use_xyz):
    if image_T1.shape != image_T2.shape:
        raise ValueError(' shape of images(T1) are not same as shape of images(T2)')
    if use_roi == 0:
        roi = np.ones([image_T1.shape[0], image_T1.shape[1], image_T1.shape[2]], dtype=int)
    if skull_strip == 1:
        image_T1 = np.multiply(image_T1, roi)
        image_T2 = np.multiply(image_T2, roi)

    sample_coords = []
    sample_list = []
    zmin_next = 0
    z_stop = False

    while not z_stop:
        zmax = min(zmin_next + sample_size, image_T1.shape[2])
        zmin = zmax - sample_size
        zmin_next = zmin_next + step_size
        if zmax < image_T1.shape[2]:
            z_stop = False
        else:
            z_stop = True

        ymin_next = 0
        y_stop = False
        while not y_stop:
            ymax = min(ymin_next + sample_size, image_T1.shape[1])
            ymin = ymax - sample_size
            ymin_next = ymin_next + step_size
            if ymax < image_T1.shape[1]:
                y_stop = False
            else:
                y_stop = True

            xmin_next = 0

            # print(xmin_next)
            x_stop = False
            while not x_stop:
                # print(xmin_next)
                xmax = min(xmin_next + sample_size, image_T1.shape[0])
                xmin = xmax - sample_size
                xmin_next = xmin_next + step_size
                if xmax < image_T1.shape[0]:
                    x_stop = False
                else:
                    x_stop = True

                if isinstance(roi, np.ndarray):
                    if not np.any(roi[xmin:xmax, ymin:ymax, zmin:zmax]):
                        continue

                sample_coords.append([xmin, ymin, zmin])
                if use_xyz == 1:
                    image_sample_data_T1 = image_T1[xmin:xmax, ymin: ymax, zmin:zmax]
                    image_sample_data_T2 = image_T2[xmin:xmax, ymin: ymax, zmin:zmax]
                    image_sample_data = image_sample_data_T1
                    # get coordinates
                    image_sample_x = np.asarray(list(range(xmin, xmax)))
                    image_sample_y = np.asarray(list(range(ymin, ymax)))
                    image_sample_z = np.asarray(list(range(zmin, zmax)))
                    # change shape, so easy to append rows or columns
                    image_sample_x = np.reshape(image_sample_x, len(list(range(xmin, xmax))))
                    image_sample_y = np.reshape(image_sample_y, len(list(range(ymin, ymax))))
                    image_sample_z = np.reshape(image_sample_z, len(list(range(zmin, zmax))))
                    x_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    y_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    z_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    for i_x in range(0, len(range(xmin, xmax))):
                        x_coor[i_x, :, :] = image_sample_x[i_x]
                    for i_y in range(0, len(range(ymin, ymax))):
                        y_coor[:, i_y, :] = image_sample_y[i_y]
                    for i_z in range(0, len(range(zmin, zmax))):
                        z_coor[:, :, i_z] = image_sample_z[i_z]

                    image_sample_data = np.expand_dims(image_sample_data, axis=0)
                    x_coor = np.expand_dims(x_coor, axis=0)
                    y_coor = np.expand_dims(y_coor, axis=0)
                    z_coor = np.expand_dims(z_coor, axis=0)
                    image_sample_data_T2 = np.expand_dims(image_sample_data_T2, axis=0)
                    image_sample = np.concatenate((image_sample_data, image_sample_data_T2, x_coor, y_coor, z_coor), axis=0)
                    sample_list.append(image_sample)
                else:
                    image_sample = np.concatenate(
                        (np.expand_dims(image_T1[xmin:xmax, ymin: ymax, zmin:zmax], axis=0),
                        np.expand_dims(image_T2[xmin:xmax, ymin: ymax, zmin:zmax], axis=0)),
                        axis=0)
                    sample_list.append(image_sample)
    return sample_coords, sample_list




def sample_image_T1_T2_separate(image_T1, image_T2, sample_size, step_size, roi, use_roi, skull_strip, use_xyz):
    if image_T1.shape != image_T2.shape:
        raise ValueError(' shape of images(T1) are not same as shape of images(T2)')
    if use_roi == 0:
        roi = np.ones([image_T1.shape[0], image_T1.shape[1], image_T1.shape[2]], dtype=int)
    if skull_strip == 1:
        image_T1 = np.multiply(image_T1, roi)
        image_T2 = np.multiply(image_T2, roi)

    sample_coords = []
    sample_list = []
    sample_list_t2 = []
    zmin_next = 0
    z_stop = False

    while not z_stop:
        zmax = min(zmin_next + sample_size, image_T1.shape[2])
        zmin = zmax - sample_size
        zmin_next = zmin_next + step_size
        if zmax < image_T1.shape[2]:
            z_stop = False
        else:
            z_stop = True

        ymin_next = 0
        y_stop = False
        while not y_stop:
            ymax = min(ymin_next + sample_size, image_T1.shape[1])
            ymin = ymax - sample_size
            ymin_next = ymin_next + step_size
            if ymax < image_T1.shape[1]:
                y_stop = False
            else:
                y_stop = True

            xmin_next = 0

            # print(xmin_next)
            x_stop = False
            while not x_stop:
                # print(xmin_next)
                xmax = min(xmin_next + sample_size, image_T1.shape[0])
                xmin = xmax - sample_size
                xmin_next = xmin_next + step_size
                if xmax < image_T1.shape[0]:
                    x_stop = False
                else:
                    x_stop = True

                if isinstance(roi, np.ndarray):
                    if not np.any(roi[xmin:xmax, ymin:ymax, zmin:zmax]):
                        continue

                sample_coords.append([xmin, ymin, zmin])
                if use_xyz == 1:
                    image_sample_data_T1 = image_T1[xmin:xmax, ymin: ymax, zmin:zmax]
                    image_sample_data_T2 = image_T2[xmin:xmax, ymin: ymax, zmin:zmax]
                    image_sample_data = image_sample_data_T1
                    # get coordinates
                    image_sample_x = np.asarray(list(range(xmin, xmax)))
                    image_sample_y = np.asarray(list(range(ymin, ymax)))
                    image_sample_z = np.asarray(list(range(zmin, zmax)))
                    # change shape, so easy to append rows or columns
                    image_sample_x = np.reshape(image_sample_x, len(list(range(xmin, xmax))))
                    image_sample_y = np.reshape(image_sample_y, len(list(range(ymin, ymax))))
                    image_sample_z = np.reshape(image_sample_z, len(list(range(zmin, zmax))))
                    x_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    y_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    z_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    for i_x in range(0, len(range(xmin, xmax))):
                        x_coor[i_x, :, :] = image_sample_x[i_x]
                    for i_y in range(0, len(range(ymin, ymax))):
                        y_coor[:, i_y, :] = image_sample_y[i_y]
                    for i_z in range(0, len(range(zmin, zmax))):
                        z_coor[:, :, i_z] = image_sample_z[i_z]

                    image_sample_data = np.expand_dims(image_sample_data, axis=0)
                    x_coor = np.expand_dims(x_coor, axis=0)
                    y_coor = np.expand_dims(y_coor, axis=0)
                    z_coor = np.expand_dims(z_coor, axis=0)
                    image_sample_data_T2 = np.expand_dims(image_sample_data_T2, axis=0)
                    image_sample = np.concatenate((image_sample_data, x_coor, y_coor, z_coor), axis=0)
                    image_sample_t2 = np.concatenate((image_sample_data_T2, x_coor, y_coor, z_coor), axis=0)
                    sample_list.append(image_sample)
                    sample_list_t2.append(image_sample_t2)

                else:
                    image_sample = image_T1[xmin:xmax, ymin: ymax, zmin:zmax]

                    image_sample_t2 = image_T2[xmin:xmax, ymin: ymax, zmin:zmax]
                    sample_list.append(image_sample)
                    sample_list_t2.append(image_sample_t2)
    return sample_coords, sample_list, sample_list_t2


def sample_image_t1_t1t2_separate(image_T1, image_T2, sample_size, step_size, roi, use_roi, skull_strip, use_xyz):
    if image_T1.shape != image_T2.shape:
        raise ValueError(' shape of images(T1) are not same as shape of images(T2)')
    if use_roi == 0:
        roi = np.ones([image_T1.shape[0], image_T1.shape[1], image_T1.shape[2]], dtype=int)
    if skull_strip == 1:
        image_T1 = np.multiply(image_T1, roi)
        image_T2 = np.multiply(image_T2, roi)

    sample_coords = []
    sample_list = []
    sample_list_t2 = []
    zmin_next = 0
    z_stop = False

    while not z_stop:
        zmax = min(zmin_next + sample_size, image_T1.shape[2])
        zmin = zmax - sample_size
        zmin_next = zmin_next + step_size
        if zmax < image_T1.shape[2]:
            z_stop = False
        else:
            z_stop = True

        ymin_next = 0
        y_stop = False
        while not y_stop:
            ymax = min(ymin_next + sample_size, image_T1.shape[1])
            ymin = ymax - sample_size
            ymin_next = ymin_next + step_size
            if ymax < image_T1.shape[1]:
                y_stop = False
            else:
                y_stop = True

            xmin_next = 0

            # print(xmin_next)
            x_stop = False
            while not x_stop:
                # print(xmin_next)
                xmax = min(xmin_next + sample_size, image_T1.shape[0])
                xmin = xmax - sample_size
                xmin_next = xmin_next + step_size
                if xmax < image_T1.shape[0]:
                    x_stop = False
                else:
                    x_stop = True

                if isinstance(roi, np.ndarray):
                    if not np.any(roi[xmin:xmax, ymin:ymax, zmin:zmax]):
                        continue

                sample_coords.append([xmin, ymin, zmin])
                if use_xyz == 1:
                    image_sample_data_T1 = image_T1[xmin:xmax, ymin: ymax, zmin:zmax]
                    image_sample_data_T2 = image_T2[xmin:xmax, ymin: ymax, zmin:zmax]
                    image_sample_data = image_sample_data_T1
                    # get coordinates
                    image_sample_x = np.asarray(list(range(xmin, xmax)))
                    image_sample_y = np.asarray(list(range(ymin, ymax)))
                    image_sample_z = np.asarray(list(range(zmin, zmax)))
                    # change shape, so easy to append rows or columns
                    image_sample_x = np.reshape(image_sample_x, len(list(range(xmin, xmax))))
                    image_sample_y = np.reshape(image_sample_y, len(list(range(ymin, ymax))))
                    image_sample_z = np.reshape(image_sample_z, len(list(range(zmin, zmax))))
                    x_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    y_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    z_coor = np.zeros((image_sample_data.shape[0], image_sample_data.shape[1], image_sample_data.shape[2]))
                    for i_x in range(0, len(range(xmin, xmax))):
                        x_coor[i_x, :, :] = image_sample_x[i_x]
                    for i_y in range(0, len(range(ymin, ymax))):
                        y_coor[:, i_y, :] = image_sample_y[i_y]
                    for i_z in range(0, len(range(zmin, zmax))):
                        z_coor[:, :, i_z] = image_sample_z[i_z]

                    image_sample_data = np.expand_dims(image_sample_data, axis=0)
                    x_coor = np.expand_dims(x_coor, axis=0)
                    y_coor = np.expand_dims(y_coor, axis=0)
                    z_coor = np.expand_dims(z_coor, axis=0)
                    image_sample_data_T2 = np.expand_dims(image_sample_data_T2, axis=0)
                    image_sample = np.concatenate((image_sample_data, x_coor, y_coor, z_coor), axis=0)
                    image_sample_t2 = np.concatenate((image_sample_data, image_sample_data_T2, x_coor, y_coor, z_coor), axis=0)
                    sample_list.append(image_sample)
                    sample_list_t2.append(image_sample_t2)

                else:
                    #image_sample = image_T1[xmin:xmax, ymin: ymax, zmin:zmax]

                    #image_sample_t2 = image_T2[xmin:xmax, ymin: ymax, zmin:zmax]
                    image_sample = np.expand_dims(image_T1[xmin:xmax, ymin: ymax, zmin:zmax], axis=0)
                    image_sample_t2 = np.concatenate((np.expand_dims(image_T1[xmin:xmax, ymin: ymax, zmin:zmax], axis=0),
                                                      np.expand_dims(image_T2[xmin:xmax, ymin: ymax, zmin:zmax], axis=0)), axis=0)
                    sample_list.append(image_sample)
                    sample_list_t2.append(image_sample_t2)
    return sample_coords, sample_list, sample_list_t2