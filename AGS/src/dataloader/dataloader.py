from torch.utils.data import Dataset
import os
import nibabel as nib
import torch
import numpy as np
from AGS.src.Utils import Utils
import random
import torchio
from random import randint
import pandas
import SimpleITK as sitk

class SubcorticalDataset(Dataset):

    def __init__(self, image_dir, label_dir, roi_dir, nClass,
                 mode='train', skull_stripe=True, transform=None, classification=False, canny=False):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.roi_dir = roi_dir
        self.x_data_list = sorted(os.listdir(self.image_dir))
        self.y_data_list = sorted(os.listdir(self.label_dir))
        self.roi_list = sorted(os.listdir(roi_dir))
        self.nClass = nClass
        self.mode = mode
        self.skull_stripe = skull_stripe
        self.transform = transform

        self.classification = classification
        self.canny = canny


    def __getitem__(self, index):

        self.x_data = self.x_data_list[index]
        self.y_data = self.y_data_list[index]
        self.roi_data = self.roi_list[index]
        image, image_data = Utils.load_nii(self.image_dir, self.x_data)
        label, label_data = Utils.load_nii(self.label_dir, self.y_data)
        roi, roi_data = Utils.load_nii(self.roi_dir, self.roi_data)


        if len(image.shape) > 3:
            image = np.squeeze(image, axis=3)
            label = np.squeeze(label, axis=3)
            roi = np.squeeze(roi, axis=3)
        # in case label is not numpy, this is the rare case
        label = np.asarray(label)
        # import time
        # t = time.time()
        if self.transform is not None:
            # apply = bool(random.getrandbits(1))
            a = random.random()
            if a > 0.5:
                apply = True
                # print('apply')
            else:
                apply = False
                # print('not apply')
            if apply:
                # for i in range(0,99):
                #random_number = np.random.randint(4)
                random_number = randint(0, 2)
                # random_number = 1
                # print(random_number)
                # image, label, roi = self.apply_transform(random_number,
                #                                          self.image_dir, self.x_data,
                #                                          self.label_dir, self.y_data,
                #                                          self.roi_dir, self.roi_data)
                image, label, roi = self.apply_transform(random_number,
                                                         self.image_dir, self.x_data,
                                                         self.label_dir, self.y_data,
                                                         self.roi_dir, self.roi_data)
        # print(time.time() - t)


        if self.skull_stripe:
            image = np.multiply(image, roi)
        else:
            image = image

        #image = np.expand_dims(image, axis=0)

        label_output = np.zeros((self.nClass, label.shape[0], label.shape[1], label.shape[2]), dtype=int)
        for i in range(0, self.nClass):
            if i == 0:
                label_output[i, :, :, :] = np.ones(label.shape)
            else:
                temp_labels = np.zeros(label.shape)
                temp_labels[label == i] = 1
                label_output[i, :, :, :] = temp_labels
                label_output[0, :, :, :] = label_output[0, :, :, :] - label_output[i, :, :, :]

        image = image.astype(float)
        label = label.astype(int)

        if self.classification:
            c_label = self.find_classification_label(self.x_data)



        if self.canny:
            # edge = self.get_canny_edge(self.image_dir, self.x_data)
            edge = self.get_canny_edge1(image)
            edge = np.expand_dims(edge, axis=0)
            image = np.expand_dims(image, axis=0)
            image = np.concatenate((image, edge), axis=0)

        else:
            image = np.expand_dims(image, axis=0)




        if self.mode == 'train':
            if self.classification:
                return image, label_output, label, c_label
            else:
                return image, label_output, label

        if self.mode == 'validation':
            # label_name is the segmentation name to be saved
            label_dict = {'label_name': self.y_data,
                          'label_dir': self.label_dir,
                          'label_num_total': len(self.y_data_list),
                          'image_name': self.x_data,
                          'image_dir': self.image_dir}
            if self.classification:
                return image, label_output, label, c_label, label_dict
            else:
                return image, label_output, label, label_dict

    def __len__(self):
        return len(self.x_data_list)

    def find_classification_label(self, image_name, hm=True):
        df = pandas.read_csv('/home/hao/Hao/Data/PREDICT_DATA/LogismosBSubjects_Info.csv')
        col_scanid = df['scanid']
        col_external_id = df['external_id']
        col_cap_d_group = df['cap_d_group']
        classification_label_value = 0

        if hm:
            image_name_split = image_name.split('_')
            external_id = image_name_split[1]
            scan_id = image_name_split[3]
            for j in range(0, len(col_external_id)):
                if col_external_id[j] == external_id and col_scanid[j] == int(scan_id):
                    hd = col_cap_d_group[j]

                    if hd == 'cont':
                        classification_label_value = 0
                    elif hd == 'low':
                        classification_label_value = 1
                    elif hd == 'med':
                        classification_label_value = 2
                    else:
                        classification_label_value = 3
        return classification_label_value


    def get_canny_edge(self, image_dir, image_name, lowerThreshold=50, upperThreshold=200, variance=[3] * 3):
        path_image = os.path.join(image_dir, image_name)
        image_sitk = sitk.ReadImage(path_image)

        edges = sitk.CannyEdgeDetection(image_sitk, lowerThreshold=lowerThreshold, upperThreshold=upperThreshold,
                                        variance=variance)
        edges_array = sitk.GetArrayFromImage(edges)

        return edges_array
    def get_canny_edge1(self, image, lowerThreshold=50, upperThreshold=200, variance=[3] * 3):

        image_sitk = sitk.GetImageFromArray(image)

        edges = sitk.CannyEdgeDetection(image_sitk, lowerThreshold=lowerThreshold, upperThreshold=upperThreshold,
                                        variance=variance)
        edges_array = sitk.GetArrayFromImage(edges)

        return edges_array



    def apply_transform(self, random_number, image_dir, image_name, label_dir, label_name, roi_dir, roi_name):
        path_image = os.path.join(image_dir, image_name)
        path_label = os.path.join(label_dir, label_name)
        path_roi = os.path.join(roi_dir, roi_name)
        subject_dict = {
            'image': torchio.Image(path_image, torchio.INTENSITY),
            'label': torchio.Image(path_label, torchio.LABEL),
            'roi': torchio.Image(path_roi, torchio.LABEL)
        }
        subjects = []
        subject = torchio.Subject(subject_dict)
        subjects.append(subject)
        dataset = torchio.SubjectsDataset(subjects)
        if random_number != 2:
            transform = self.transform.transforms[random_number]
            # print(self.transform.transform.transforms[random_number])
        else:
            transform = self.transform
            # print(self.transform)
        transform_dict = transform(dataset[0])

        transform_image = transform_dict['image']['data'].data.numpy()
        transform_image = np.squeeze(transform_image, axis=0)

        transform_label = transform_dict['label']['data'].data.numpy()
        transform_label = np.squeeze(transform_label, axis=0)

        transform_roi = transform_dict['roi']['data'].data.numpy()
        transform_roi = np.squeeze(transform_roi, axis=0)
        # Utils.save_nii(np.multiply(transform_image, transform_roi), image_data, os.getcwd(), image_name)
        # Utils.save_nii(transform_label, label_data, os.getcwd(), label_name)
        # Utils.save_nii(transform_roi, roi_data, os.getcwd(), roi_name)
        return transform_image, transform_label, transform_roi

    def apply_transform3(self, random_number, image_dir, image_name,
                         image_dir_GA, image_name_GA,
                         image_dir_GB, image_name_GB,
                         label_dir, label_name, roi_dir, roi_name):
        path_image = os.path.join(image_dir, image_name)
        path_label = os.path.join(label_dir, label_name)
        path_roi = os.path.join(roi_dir, roi_name)

        path_image_GA = os.path.join(image_dir_GA, image_name_GA)
        path_image_GB = os.path.join(image_dir_GB, image_name_GB)
        subject_dict = {
            'image': torchio.Image(path_image, torchio.INTENSITY),
            'image_GA': torchio.Image(path_image_GA, torchio.INTENSITY),
            'image_GB': torchio.Image(path_image_GB, torchio.INTENSITY),
            'label': torchio.Image(path_label, torchio.LABEL),
            'roi': torchio.Image(path_roi, torchio.LABEL)
        }
        subjects = []
        subject = torchio.Subject(subject_dict)
        subjects.append(subject)
        dataset = torchio.ImagesDataset(subjects)
        if random_number != 2:
            transform = self.transform.transform.transforms[random_number]
            # print(self.transform.transform.transforms[random_number])
        else:
            transform = self.transform
            # print(self.transform)
        transform_dict = transform(dataset[0])

        transform_image = transform_dict['image']['data'].data.numpy()
        transform_image = np.squeeze(transform_image, axis=0)

        transform_label = transform_dict['label']['data'].data.numpy()
        transform_label = np.squeeze(transform_label, axis=0)

        transform_roi = transform_dict['roi']['data'].data.numpy()
        transform_roi = np.squeeze(transform_roi, axis=0)

        transform_image_GA = transform_dict['image_GA']['data'].data.numpy()
        transform_image_GA = np.squeeze(transform_image_GA, axis=0)

        transform_image_GB = transform_dict['image_GB']['data'].data.numpy()
        transform_image_GB = np.squeeze(transform_image_GB, axis=0)
        # Utils.save_nii(np.multiply(transform_image, transform_roi), image_data, os.getcwd(), image_name)
        # Utils.save_nii(transform_label, label_data, os.getcwd(), label_name)
        # Utils.save_nii(transform_roi, roi_data, os.getcwd(), roi_name)
        return transform_image, transform_image_GA, transform_image_GB, transform_label, transform_roi,
