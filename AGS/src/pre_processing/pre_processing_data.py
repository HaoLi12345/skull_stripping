import os
import SimpleITK as sitk
import ants
import numpy as np
import nibabel as nib
import vtk


def registration(image_dir, template_path, saving_dir):
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    fi = ants.image_read(template_path)
    image_list = sorted(os.listdir(image_dir))
    for name in image_list:
        image_path = os.path.join(image_dir, name)
        mi = ants.image_read(image_path)
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform='Rigid')
        saved_image = mytx['warpedmovout']
        save_path = os.path.join(saving_dir, name)
        ants.image_write(saved_image, save_path)


def registration_HD(image_dir, roi_dir, template_path,
                    saving_image_dir, saving_roi_dir, saving_transform_dir,
                    type_of_transform='Rigid'):

    if not os.path.exists(saving_image_dir):
        os.mkdir(saving_image_dir)
    if not os.path.exists(saving_roi_dir):
        os.mkdir(saving_roi_dir)
    if not os.path.exists(saving_transform_dir):
        os.mkdir(saving_transform_dir)

    fi = ants.image_read(template_path)
    image_list = sorted(os.listdir(image_dir))
    roi_list = sorted(os.listdir(roi_dir))
    for i in range(0, len(image_list)):
        image_path = os.path.join(image_dir, image_list[i])
        mi = ants.image_read(image_path)
        mytx = ants.registration(fixed=fi, moving=mi, type_of_transform=type_of_transform)

        # transformation
        forward_transform = mytx['fwdtransforms']
        name = image_list[i]
        trans_name = name.split('.')[0]
        save_transform_path = os.path.join(saving_transform_dir, 'forward_transform_' + trans_name + '.mat')
        fwd_trans = ants.read_transform(forward_transform[0])
        ants.write_transform(fwd_trans, save_transform_path)

        # image
        saved_image = mytx['warpedmovout']
        save_path = os.path.join(saving_image_dir, image_list[i])
        ants.image_write(saved_image, save_path)

        # roi
        roi_path = os.path.join(roi_dir, roi_list[i])
        roi = ants.image_read(roi_path)
        saved_roi = ants.apply_transforms(fixed=saved_image, moving=roi, interpolator='nearestNeighbor',
                                          transformlist=mytx['fwdtransforms'])
        save_path_roi = os.path.join(saving_roi_dir, roi_list[i])
        ants.image_write(saved_roi, save_path_roi)
        # print('roi saved')

        # original_image = ants.apply_transforms(fixed=mi, moving=saved_image, transformlist=forward_transform, whichtoinvert=[1])
        # ants.image_write(original_image, os.path.join(os.getcwd(), '1.nii.gz'))
        #
        # original_roi = ants.apply_transforms(fixed=roi, moving=saved_roi, interpolator='nearestNeighbor', whichtoinvert=[1],
        #                                  transformlist=forward_transform)
        # ants.image_write(original_roi, os.path.join(os.getcwd(), '2.nii.gz'))
        print(1)



def N4(image_dir, saving_image_dir):
    if not os.path.exists(saving_image_dir):
        os.mkdir(saving_image_dir)
    image_list = sorted(os.listdir(image_dir))
    for name in image_list:
        image_path = os.path.join(image_dir, name)
        inputImage = sitk.ReadImage(image_path)
        maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
        inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        output = corrector.Execute(inputImage, maskImage)

        save_path = os.path.join(saving_image_dir, name)
        sitk.WriteImage(output, save_path)


def image_resample(template_path, save_name, x, y, z, xs, ys, zs,
                   mode, saving_dir, change_spacing=True, auto_spacing=True, x_off=0, y_off=0, z_off=0):
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    img = sitk.ReadImage(template_path)

    original_spacing = img.GetSpacing()

    print('original_spacing:', original_spacing)

    original_size = img.GetSize()
    print('original_size:', original_size)
    if change_spacing:
        factor1 = x / img.GetSize()[0]

        factor2 = y / img.GetSize()[1]

        factor3 = z / img.GetSize()[2]

        factor = [factor1, factor2, factor3]

        new_spacing = np.asarray(img.GetSpacing()) / factor
    else:
        new_spacing = original_spacing

    if not auto_spacing:
        new_spacing = (xs, ys, zs)


    new_size = [x, y, z]
    translation = sitk.TranslationTransform(3)

    x_trans = round((x - img.GetSize()[0]) / 2) - x_off
    y_trans = round((y - img.GetSize()[1]) / 2) - y_off
    z_trans = round((z - img.GetSize()[2]) / 2) - z_off
    translation.SetOffset((-x_trans, -y_trans, -z_trans))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)

    resampler.SetTransform(translation)

    resampler.SetOutputDirection = img.GetDirection()
    resampler.SetOutputOrigin = img.GetOrigin()
    #
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)

    if mode == 'image':
        interpolator = sitk.sitkLinear
        resampler.SetInterpolator(interpolator)
    if mode == 'label':
        interpolator = sitk.sitkNearestNeighbor
        resampler.SetInterpolator(interpolator)

    imgResampled = resampler.Execute(img)

    new_spacing = imgResampled.GetSpacing()

    print('new_spacing:', new_spacing)
    print('new_size:', imgResampled.GetSize())

    print('')

    sitk.WriteImage(imgResampled, os.path.join(saving_dir, save_name))
    # imgResampled_array = sitk.GetArrayFromImage(imgResampled)
    # return imgResampled_array


def readnrrd(filename):
    """Read image in nrrd format."""
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(filename)
    reader.Update()
    info = reader.GetInformation()
    return reader.GetOutput(), info


def writenifti(image,filename, info):
    """Write nifti file."""
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(image)
    writer.SetFileName(filename)
    writer.SetInformation(info)
    writer.Write()




if __name__ == '__main__':

    resample_template_path = '/home/hao/Hao/AGS/Dataset/template_resample.nii.gz'
    image_dir = '/home/hao/Hao/AGS/Dataset/3D_T1_pre_processing/3D_T1_all_image'
    saving_dir1 = '/home/hao/Hao/AGS/Dataset/3D_T1_pre_processing/registration1'
    registration(image_dir, resample_template_path, saving_dir1)

    # saving_dir2 = '/home/hao/Hao/AGS/Dataset/3D_T1_pre_processing/N4'
    # N4(saving_dir1, saving_dir2)
    #
    # saving_dir3 = '/home/hao/Hao/AGS/Dataset/3D_T1_pre_processing/registration2'
    # registration(saving_dir2, resample_template_path, saving_dir3)


