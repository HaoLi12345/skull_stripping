import os
import random
from AGS.src.Utils import Utils
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
from AGS.src.dataloader import dataloader
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from AGS.src.model import subcortical_seg_model
from AGS.src.model import vanila_3dunet
from AGS.src.model.subcortical_seg_model import SingleConv, DoubleConv, ExtResNetBlock
from AGS.src.losses import focal_loss
from AGS.src.losses import dice_loss
from torchsummary import summary
import torchio
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


def get_loss_function_weights(label_dir, nClass):
    label_list = sorted(os.listdir(label_dir))
    counts_total = np.zeros(nClass, dtype=int)
    for i in range(0, len(label_list)):
        label, _ = Utils.load_nii(label_dir, label_list[i])
        unique, counts = np.unique(label, return_counts=True)
        if counts_total.shape != counts.shape:
            raise ValueError('labels have more than ' + str(nClass) + ' classes')
        counts_total += counts
    sum_counts_total = counts_total.sum()
    weights = sum_counts_total / counts_total
    weights = torch.tensor(weights)
    weights = weights.float().cuda()
    return weights


def initialization(image_dir, label_dir, roi_dir,
                   validation_image_dir, validation_label_dir, validation_roi_dir,
                   nClass, basic_module_encoder=DoubleConv, basic_module_decoder=DoubleConv, weights=False, classification=False, canny=False, vanila=False):
    if canny:
        if vanila:
            model = vanila_3dunet.Unet3d_vanila(2, nClass, f_maps=[64, 128, 256, 256])
        else:
            model = subcortical_seg_model.Unet3d(2, nClass, layer_order='cbr',
                                                 basic_module_encoder=basic_module_encoder,
                                                 basic_module_decoder=basic_module_decoder,
                                                 num_levels=5,
                                         f_maps=[16, 32, 64, 64, 128], classification=classification)
    else:
        if vanila:
            model = vanila_3dunet.Unet3d_vanila(1, nClass, f_maps=16)
        else:
            model = subcortical_seg_model.Unet3d(1, nClass, layer_order='cbr',
                                                 basic_module_encoder=basic_module_encoder,
                                                 basic_module_decoder=basic_module_decoder,
                                                 num_levels=5,
                                             f_maps=[16, 32, 64, 64, 128], classification=classification)
    model.cuda()
    # summary(model, tuple([1, 128, 128, 96]))
    print(model)

    loss_criterion_focal = focal_loss.FocalLoss_Ori(num_class=nClass, alpha=0.1,
                                              gamma=2.0, balance_index=0)

    loss_criterion_dice = dice_loss.DiceLoss(weight=torch.tensor([0.1, 1]).cuda(), sigmoid_normalization=False)
    # loss_criterion_dice = dice_loss.GeneralizedDiceLoss(sigmoid_normalization=False)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.00001)

    transform = torchio.transforms.Compose([torchio.transforms.RandomAffine(scales=(0.9, 1.2), degrees=(10), isotropic=False,default_pad_value='otsu',image_interpolation=torchio.transforms.Interpolation.BSPLINE),
                                            torchio.transforms.RandomElasticDeformation(num_control_points=7,
                                                                                        max_displacement=6,
                                                                                        locked_borders=2,)])

    dataset = dataloader.SubcorticalDataset(image_dir,
                                            label_dir, roi_dir, nClass, skull_stripe=False,
                                            transform=transform, classification=classification, canny=canny)
    train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=4)

    dataset_validation = dataloader.SubcorticalDataset(validation_image_dir,
                                                       validation_label_dir, validation_roi_dir,
                                                       nClass, skull_stripe=False,
                                                       mode='validation', transform=None, classification=classification,
                                                       canny=canny)

    validation_loader = DataLoader(dataset=dataset_validation, batch_size=1, shuffle=False, num_workers=4)

    return model, loss_criterion_focal, loss_criterion_dice, optimizer, train_loader, validation_loader


def set_scheduler(optimizer, step_size=50, gamma=0.5):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return scheduler


def validation(model, validation_loader, loss_criterion_BCE, loss_criterion_dice, validation_saving_dir, nClass=9,
               classfication=False):
    # TODO check validation batch size, make multi-batch size version
    model.eval()
    dice = np.zeros(nClass - 1)
    accuracy = 0
    number = 0
    with torch.no_grad():

        validation_loss_all_image = 0.0

        for i, data in enumerate(validation_loader, 0):
            # labels_output is multi-channel version of label
            # ex: label size in b,x,y,z contains n classes
            # labels_output size in b,n,x,y,z
            if classification:
                inputs, labels_output, labels, c_labels, labels_dict = data
                inputs = inputs.float().cuda()
                labels = labels.long().cuda()
                # labels_output is used for multi-channel losses
                labels_output = labels_output.float().cuda()
                c_labels = c_labels.long().cuda()
                #c_labels = c_labels.view(-1, 1)
                outputs, c_outputs = model(inputs)

                loss = loss_criterion_dice(outputs, labels_output) + loss_criterion_BCE(c_outputs, c_labels)
                _, c_predicted = torch.max(c_outputs.data, 1)

                print('groundtruth/predicted  {}/{}: '.format(c_labels.item(), c_predicted.item()))

                if c_labels == c_predicted:
                    number = number + 1
                accuracy = number / labels_dict['label_num_total'][0].item()
            else:
                inputs, labels_output, labels, labels_dict = data
                inputs = inputs.float().cuda()
                labels = labels.long().cuda()
                # labels_output is used for multi-channel losses
                labels_output = labels_output.float().cuda()

                # outputs1, outputs = model(inputs)
                outputs = model(inputs)

                loss = loss_criterion_dice(outputs, labels_output)
            _, predicted = torch.max(outputs.data, 1)

            # outputs1 = F.softmax(outputs1, dim=1)
            # outputs = F.softmax(outputs, dim=1)
            # loss = loss_criterion(outputs, labels_output) + loss_criterion(outputs1, labels_output)

            # loss = loss_criterion(outputs, labels)


            validation_loss_all_image += loss.item()

            numpy_predicted = predicted.cpu().numpy()
            numpy_predicted = np.squeeze(numpy_predicted, axis=0)
            numpy_label = labels.cpu().numpy()
            numpy_label = np.squeeze(numpy_label, axis=0)

            label_name = labels_dict['label_name'][0]
            label_dir = labels_dict['label_dir'][0]
            image_name = labels_dict['image_name'][0]
            _, label_data = Utils.load_nii(label_dir, label_name)
            Utils.save_nii(numpy_predicted, label_data, validation_saving_dir, 'Segmentation_' + image_name)
            number = number + 1
            dice_array = Utils.computeDice(numpy_predicted, numpy_label)
            dice = dice + np.asarray(dice_array)
            print(number)
            for d_i in range(len(dice_array)):
                print(" -------------- DSC (Class {}) : {}".format(str(d_i + 1), dice_array[d_i]))
            print('')

    validation_loss = (validation_loss_all_image / labels_dict['label_num_total'][0].item())
    dice_avg = dice / labels_dict['label_num_total'][0].item()
    print(accuracy)
    return validation_loss, dice_avg, accuracy


def early_stop(best_dice, current_dice, index, stop_epoch=20):
    print('previous dice (average for all validation images):' + str(best_dice))
    print('current  dice (average for all validation images):' + str(current_dice))

    early_stop_condition = False
    save_model_condition = True
    if best_dice.sum() <= current_dice.sum():
        print('best dice replaced by current dice')
        best_dice = current_dice
        index = 0
    elif best_dice.sum() > current_dice.sum():
        print('best dice replaced by previous dice')
        best_dice = best_dice
        index = index + 1
        save_model_condition = False
    if index == stop_epoch:
        early_stop_condition = True

    return early_stop_condition, save_model_condition, best_dice, index


def plot_loss(training_loss, validation_loss):
    plt.plot(np.array(training_loss), 'r', label='training_loss')
    plt.plot(np.array(validation_loss), 'b', label='validation_loss')
    ax = plt.gca()
    ax.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def set_seed():
    seed = 10
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # make cudnn to be reproducible for performance
    # can be commented for faster training
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def trainer(model, model_saving_dir, loss_criterion_BCE, loss_criterion_dice, optimizer,
            train_loader, validation_loader, scheduler, validation_saving_dir, nClass=9, total_epoch_num=100,
            classification=False, resume=False):
    t = time.time()
    training_loss = []
    validation_loss = []
    best_dice = np.zeros(nClass - 1)
    stop_index = 0
    early_stop_condition = False
    save_model_condition = True

    if resume:
        checkpoint = torch.load(os.path.join(model_saving_dir, 'best_validation.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loss = checkpoint['loss']
        best_dice = checkpoint['best_dice']
        stop_index = checkpoint['stop_index']


    else:
        current_epoch = -1
        best_dice = np.zeros(nClass - 1)
        stop_index = 0
    for epoch in range(current_epoch + 1, total_epoch_num):  # loop over the dataset multiple times

        model.train()
        t_epoch = time.time()
        training_loss_epoch = 0.0
        loss = 0
        for i, data in enumerate(train_loader, 0):
            if classification:
                inputs, labels_output, labels, c_labels = data
                inputs = inputs.float().cuda()
                labels_output = labels_output.float().cuda()
                labels_output = labels_output.long().cuda()
                labels = labels.long().cuda()
                c_labels = c_labels.long().cuda()
                # BCE
                # c_labels = c_labels.view(-1,)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs, c_outputs = model(inputs)


                loss = loss_criterion_dice(outputs, labels_output) + loss_criterion_BCE(c_outputs, c_labels)

                if i % 1 == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.5f ' % (epoch, i, loss,),
                          '[%d, %5d] dice_loss: %.5f ' % (epoch, i, loss_criterion_dice(outputs, labels_output),),
                          '[%d, %5d] CE_loss: %.5f ' % (epoch, i, loss_criterion_BCE(c_outputs, c_labels),),

                          )
            else:
                inputs, labels_output, labels = data
                inputs = inputs.float().cuda()
                labels_output = labels_output.float().cuda()
                labels_output = labels_output.long().cuda()
                labels = labels.long().cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = loss_criterion_dice(outputs, labels_output)
                if i % 1 == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.5f ' %
                          (epoch + 1, i + 1, loss,
                           ))

            loss.backward()
            optimizer.step()
            training_loss_epoch += loss.item()




        # update training loss
        training_loss.append(training_loss_epoch / (i + 1))


        if epoch % 1 == 0:
            validation_loss_epoch, validation_dice_avg, accuracy = validation(model, validation_loader,
                                                                    loss_criterion_BCE, loss_criterion_dice, validation_saving_dir)
            validation_loss.append(validation_loss_epoch)
            early_stop_condition, save_model_condition, best_dice, stop_index = \
                early_stop(best_dice, validation_dice_avg, stop_index, stop_epoch=int(total_epoch_num/5*5))
            print('early stop index: {}/{}'.format(stop_index, int(total_epoch_num/5*5)))
        # save the model
        print(1)
        if save_model_condition:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'stop_index': stop_index
            }, os.path.join(model_saving_dir, 'best_validation.pth'))
            # torch.save(model.state_dict(),
            #            os.path.join(model_saving_dir, 'best_validation.pth'))
            print('best validation model saved')

        if early_stop_condition:
            print('early stop triggered')
            break

        t_epoch_diff = time.time() - t_epoch
        print('epoch {}/{} spend {} seconds '.format(epoch + 1, total_epoch_num, t_epoch_diff))

        # update learning rate
        scheduler.step()
        print('Epoch {}, lr {}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
    t_total_training = time.time() - t
    print('Finished Training')
    print('total training cost {}'.format(t_total_training))

    plot_loss(training_loss, validation_loss)




def main(image_dir, label_dir, roi_dir,
         validation_image_dir,
         validation_label_dir, validation_roi_dir,
         nClass, prefix=None, classification=False, resume=False, canny=False, vanila=False):


    output_dir, model_saving_dir, prediction_dir, testing_dir, validation_dir = creater_outputs_folders(prefix=prefix)
    model, loss_criterion_BCE, loss_criterion_dice, optimizer, train_loader, validation_loader = initialization(image_dir,
                                                                                                                label_dir, roi_dir,
                                                                                       validation_image_dir,
                                                                                       validation_label_dir,
                                                                                       validation_roi_dir,
                                                                                       nClass,basic_module_encoder=ExtResNetBlock,
                                                                                       basic_module_decoder=ExtResNetBlock,
                                                                                       weights=True, classification=classification,
                                                                                                                canny=canny, vanila=vanila)
    scheduler = set_scheduler(optimizer)
    trainer(model, model_saving_dir, loss_criterion_BCE, loss_criterion_dice, optimizer, train_loader, validation_loader, scheduler,
            validation_saving_dir=validation_dir, classification=classification, resume=resume, )



if __name__ == '__main__':

    ##TODO: change image dirrections, dataloader, and input channel number -----> back to the version ISBI2021
    # HD
    image_dir = '/home/hao/Hao/AGS/Dataset/predict_hd/train_registration_image_HM/all_image'
    label_dir = '/home/hao/Hao/AGS/Dataset/predict_hd/train_registration/all_roi'
    roi_dir = label_dir

    validation_image_dir = '/home/hao/Hao/AGS/Dataset/predict_hd/validation_registration_image_HM/all_image'
    validation_label_dir = '/home/hao/Hao/AGS/Dataset/predict_hd/validation_registration/all_roi'
    validation_roi_dir = validation_label_dir

    # infant
    # image_dir = '/home/hao/Hao/AGS/Dataset/infant/train_registration/T1'
    # label_dir = '/home/hao/Hao/AGS/Dataset/infant/train_registration/mask'
    # roi_dir = label_dir
    #
    # validation_image_dir = '/home/hao/Hao/AGS/Dataset/infant/validation_registration/T1'
    # validation_label_dir = '/home/hao/Hao/AGS/Dataset/infant/validation_registration/mask'
    # validation_roi_dir = validation_label_dir


    # HD
    # image_dir = '/home/hao/Hao/AGS/Dataset/predict_hd/train_registration_image_HM/all_image'
    # image_dir_GA = '/home/hao/Hao/AGS/Dataset/predict_hd/cycleGAN/train/GA'
    # image_dir_GB = '/home/hao/Hao/AGS/Dataset/predict_hd/cycleGAN/train/GB'
    #
    # label_dir = '/home/hao/Hao/AGS/Dataset/predict_hd/train_registration/all_roi'
    # roi_dir = label_dir
    #
    # validation_image_dir = '/home/hao/Hao/AGS/Dataset/predict_hd/validation_registration_image_HM/all_image'
    # validation_image_dir_GA = '/home/hao/Hao/AGS/Dataset/predict_hd/cycleGAN/validation/GA'
    # validation_image_dir_GB = '/home/hao/Hao/AGS/Dataset/predict_hd/cycleGAN/validation/GB'
    #
    # validation_label_dir = '/home/hao/Hao/AGS/Dataset/predict_hd/validation_registration/all_roi'
    # validation_roi_dir = validation_label_dir



    # TODO change lr and early stop
    classification = False
    canny = False
    vanila = False
    resume = True
    # prefix = 'HD_skull_stripping_vanila3dunet'

    # prefix = 'infant_skull_stripping_no_attention_midconv_k5'
    prefix = 'HD_skull_stripping_no_attention_midconv_k5'



    nClass = 2


    # set_seed()
    main(image_dir, label_dir, roi_dir,
         validation_image_dir,
         validation_label_dir, validation_roi_dir,
         nClass, prefix=prefix, classification=classification, resume=resume, canny=canny, vanila=vanila)
