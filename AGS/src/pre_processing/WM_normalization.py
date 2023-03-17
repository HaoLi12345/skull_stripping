import os
from src.pre_processing.intensity_normalization_github.intensity_normalization import normalize
from src.pre_processing.intensity_normalization_github.intensity_normalization import plot
import matplotlib.pyplot as plt
import shutil
import SimpleITK as sitk

def move_to_folder(src, dest):
    src_list = sorted(os.listdir(src))
    for i in range(0, len(src_list)):
        name = src_list[i]
        src_path = os.path.join(src, name)
        dest_path = os.path.join(dest, name)
        shutil.copy(src_path, dest_path)
def move_to_all(ags_dir, predict_train_dir, predict_vali_dir, predict_test_dir, all_dir):
    move_to_folder(ags_dir, all_dir)
    move_to_folder(predict_train_dir, all_dir)
    move_to_folder(predict_vali_dir, all_dir)
    move_to_folder(predict_test_dir, all_dir)


def check_name(name, hd_list):
    exist = False
    name_split = name.split('_')
    external_id = name_split[1]
    scan_id = name_split[3]
    for i in range(0, len(hd_list)):
        hd_name = hd_list[i]
        hd_name_split = hd_name.split('_')
        hd_external_id = hd_name_split[0]
        hd_scan_id = hd_name_split[2]
        if external_id == hd_external_id and int(scan_id) == int(hd_scan_id):
            exist = True
            break
    return exist
def move_from_all(all_dir, ags_dir_src, ags_dir_dest,
                  hd_train_src, hd_train_dest,
                  hd_vali_src, hd_vali_dest,
                  hd_test_src, hd_test_dest,):
    all_list = sorted(os.listdir(all_dir))
    # ags_list = sorted(os.listdir(ags_dir_src))
    hd_train_list = sorted(os.listdir(hd_train_src))
    hd_vali_list =sorted(os.listdir(hd_vali_src))
    hd_test_list = sorted(os.listdir(hd_test_src))

    for i in range(0,len(all_list)):
        name = all_list[i]
        name_split = name.split('_')
        if name_split[1] == 'LD':
            image_src_path = os.path.join(all_dir, name)
            image_dest_path = os.path.join(ags_dir_dest, name)
            shutil.copy(image_src_path, image_dest_path)
        else:

            exist_train = check_name(name, hd_train_list)
            if exist_train:
                image_src_path = os.path.join(all_dir, name)
                image_dest_path = os.path.join(hd_train_dest, name)
                shutil.copy(image_src_path, image_dest_path)

            exist_vali = check_name(name, hd_vali_list)
            if exist_vali:
                image_src_path = os.path.join(all_dir, name)
                image_dest_path = os.path.join(hd_vali_dest, name)
                shutil.copy(image_src_path, image_dest_path)

            exist_test = check_name(name, hd_test_list)
            if exist_test:
                image_src_path = os.path.join(all_dir, name)
                image_dest_path = os.path.join(hd_test_dest, name)
                shutil.copy(image_src_path, image_dest_path)


def make_mask(image_dir, mask_dir):
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    image_list = sorted(os.listdir(image_dir))
    for i in range(0, len(image_list)):
        name = image_list[i]
        image_path = os.path.join(image_dir, name)
        write_path = os.path.join(mask_dir, name)

        image = sitk.ReadImage(image_path)
        maskImage = sitk.OtsuThreshold(image, 0, 1, 200)

        close = sitk.BinaryMorphologicalClosingImageFilter()
        close.SetKernelRadius(25)
        closed = close.Execute(maskImage)

        sitk.WriteImage(closed, write_path)


shift_scale = 0
expand_scale = 1
# all
base_dir = '/home/hao/Hao/AGS_cycleGAN/Dataset'
image_dir = os.path.join(base_dir, 'image')

saving_HM = os.path.join(base_dir, 'image_HM')
if not os.path.exists(saving_HM):
    os.makedirs(saving_HM)
image_list = sorted(os.listdir(image_dir))

normalize.hm.hm_normalize(img_dir=image_dir, train_path=None,
                          mask_dir=None, output_dir=saving_HM, write_to_disk=True,
                          shift_scale=shift_scale, expand_scale=expand_scale, diff_train=False, mean_factor=1)

ax = plot.hist.all_hists(image_dir, mask_dir=None)
plt.show()
# TODO change dir
ax1 = plot.hist.all_hists(saving_HM, mask_dir=None)
plt.show()



