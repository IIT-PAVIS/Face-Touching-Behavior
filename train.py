#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
	IIT : Istituto italiano di tecnologia

    Pattern Analysis and Computer Vision (PAVIS) research line

    Usage Example:
		$ python3

    LICENSE:
	This project is licensed under the terms of the MIT license.
	This project incorporates material from the projects listed below (collectively, "Third Party Code").
	This Third Party Code is licensed to you under their original license terms.
	We reserves all other rights not expressly granted, whether by implication, estoppel or otherwise.
	The software can be freely used for any non-commercial applications.
"""

# # Importing the relevant modules
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch import nn, device, manual_seed, optim
from torchvision import transforms

from sklearn.model_selection import train_test_split
#
# from PIL import Image
# Image.MAX_IMAGE_PIXELS = 1000000000
#
import argparse
import numpy as np
import os
from os.path import isfile, join
import scipy.io
import pickle
import time
import matplotlib.pyplot

import dataset
import model
import utils

splits_dir = 'data/T1_2_3_4_5/'
# video_basedir = 'data/faces/1/'
video_basedir = 'faces4/faces/1/'

recalculate_splits = False
json_out_path = 'data/experiments.json'


def main(args):
    selected_experiment = 3
    model_path = args.output_dir + args.experiment_name + '/'

    if recalculate_splits:
        print('*** Loading the annotaion files for each dataset Splits folders ...')
        # Loading the Dataset Splits folders
        folders = [os.path.join(splits_dir, o) for o in os.listdir(splits_dir) if os.path.isdir(os.path.join(splits_dir, o))]
        folders.sort()

        # Listing all the files in the folders
        splits = []
        splits_basefolder = []
        for folder in folders:
            files = [folder+'/'+f for f in os.listdir(folder) if isfile(join(folder, f))]
            files.sort()
            splits.append(files)

        # Generating the list of files for each video split
        split_imgs = []
        split_labels = []

        print('*** Loading the frames and the annotations for each video in the split ...')
        # Loading the frames and the annotations for each video in the split
        for split in splits:
            video_imgs = []
            video_label = []

            for video_idx in range(len(split)):
                # Listing all the video images
                video_filename = os.path.basename(split[video_idx])
                video_name_comp = video_filename[:-4].split("-")

                if video_name_comp[0] == '20151001' or video_name_comp[0] == '20151006':
                    fps_name = '30'
                else:
                    fps_name = '20'

                if video_name_comp[2] == '89':
                    append_path = video_name_comp[0] + '/' + video_name_comp[1] + '/new_' + fps_name + 'fps_00408CDC17' + \
                                  video_name_comp[2] + '/'
                else:
                    append_path = video_name_comp[0] + '/' + video_name_comp[1] + '/new_' + fps_name + 'fps_00408CB749' + \
                                  video_name_comp[2] + '/'

                video_dir = video_basedir + append_path[:-1] + '_faces/'
                video_f = [video_dir+'/'+f for f in os.listdir(video_dir) if isfile(join(video_dir, f))]
                video_f.sort()

                # print('###############')
                # print('video_name_comp: {}'.format(video_name_comp))
                # print('video_dir: {}'.format(video_dir))
                # print('images found: {}'.format(len(video_f)))
                # print('mat file path: {}'.format(split[video_idx]))

                # Opening the Matlab annotation file
                annotation = scipy.io.loadmat(split[video_idx])
                label = np.squeeze(annotation['final'])

                # video_imgs.append(annotation)
                video_label.append(label)
                video_imgs.append(video_f)

                # print(' -- Labels: {} - images: {}'.format(len(video_label), len(video_imgs)))

            split_imgs.append(video_imgs)
            split_labels.append(video_label)

            # print('Labels: {} - images: {}'.format(len(split_labels), len(split_imgs)))

        print('*** Generating experiment splits ...')
        # Generating experiment splits
        experiments = []
        for exp_num in range(len(splits)):
            print('Generating experiment {}...'.format(exp_num))

            trainval_imgs = np.zeros((0))
            trainval_labels = np.zeros((0))
            test_imgs = np.zeros((0))
            test_labels = np.zeros((0))

            for split_idx in range(len(splits)):

                # print(' - split_idx {} - len: {}...'.format(split_idx, len(splits[split_idx])))

                for video_idx in range(len(splits[split_idx])):
                    # print(' -- video_idx {}...'.format(video_idx))
                    if split_idx == exp_num:
                        test_labels = np.concatenate((test_labels, split_labels[split_idx][video_idx]))
                        test_imgs = np.concatenate((test_imgs, split_imgs[split_idx][video_idx]))
                    else:
                        trainval_labels = np.concatenate((trainval_labels, split_labels[split_idx][video_idx]))
                        trainval_imgs = np.concatenate((trainval_imgs, split_imgs[split_idx][video_idx]))

            # Let's count the number of 0s and 1s in training set
            # _, counts_labels = np.unique(trainval_labels, return_counts=True)
            # print(counts_labels)
            # print("Labels' ratio: {} - Number of touches: {}".format(counts_labels[0]/counts_labels[1], counts_labels[1]))

            # Balancing the dataset
            tv_pos_imgs = trainval_imgs[trainval_labels == 1]
            tv_pos_labs = trainval_labels[trainval_labels == 1]
            tv_neg_imgs = trainval_imgs[trainval_labels == 0]
            tv_neg_labs = trainval_labels[trainval_labels == 0]

            utils.show_batch_numpy(tv_pos_imgs[:8], tv_pos_labs[:8])
            matplotlib.pyplot.show()
            utils.show_batch_numpy(tv_neg_imgs[:8], tv_neg_labs[:8])
            matplotlib.pyplot.show()

            balancing_ratio = len(tv_pos_labs)/len(tv_neg_labs)
            # print("Labels' ratio: {} - Number of touches: {}".format(balancing_ratio, len(tv_pos_labs)))

            _, tv_bal_neg_imgs, _, tv_bal_neg_labels = train_test_split(tv_neg_imgs, tv_neg_labs,
                                                                              test_size=balancing_ratio, random_state=10)

            updated_ratio = len(tv_pos_labs) / len(tv_bal_neg_labels)
            # print("Balanced ratio: {}".format(updated_ratio))
            # print("tv_bal_neg_imgs: {} - tv_bal_neg_labels: {}".format(len(tv_bal_neg_imgs), len(tv_bal_neg_labels)))

            # Regenerating TrainVal dataset to split it into Train/Val
            # print('tv_bal_neg_imgs: {} - tv_pos_imgs: {} '.format(tv_bal_neg_imgs.shape, tv_pos_imgs.shape))
            trainval_imgs = np.concatenate((tv_bal_neg_imgs, tv_pos_imgs))
            trainval_labels = np.concatenate((tv_bal_neg_labels, tv_pos_labs))
            # print("trainval_imgs: {} - tv_bal_neg_imgs*2: {}".format(len(trainval_imgs), 2*len(tv_bal_neg_imgs)))
            train_imgs, val_imgs, train_labels, val_labels = train_test_split(trainval_imgs, trainval_labels,
                                                                              test_size=0.20, random_state=10)
            experiment = {
                "train_imgs": train_imgs,
                "train_labels": train_labels,
                "val_imgs": val_imgs,
                "val_labels": val_labels,
                "test_imgs": test_imgs,
                "test_labels": test_labels,
            }
            experiments.append(experiment)

        with open(json_out_path, 'wb') as fp:
            pickle.dump(experiments, fp, protocol=pickle.HIGHEST_PROTOCOL)
            # MBMB - Verify the splits are always the same
    else:
        experiments = pickle.load(open(json_out_path, "rb"))

    # Testing Experiments[selected_experiment]
    experiment = experiments[selected_experiment]

    ## Initializng data transformations
    manual_seed(args.random_seed)

    image_size = 300
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # transforms.RandomResizedCrop(image_size)
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(img_mean, img_std)
        ]),
        'val': transforms.Compose([
            # transforms.Resize(int(1.1*image_size)),
            transforms.Resize((image_size, image_size)),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(img_mean, img_std)
        ]),
    }

    ## Initializng pyTorch dataloaders
    train_dict = {
        "imgs": experiment['train_imgs'],
        "labels": experiment['train_labels']
    }

    val_dict = {
        "imgs": experiment['val_imgs'],
        "labels": experiment['val_labels']
    }

    test_dict = {
        "imgs": experiment['test_imgs'],
        "labels": experiment['test_labels']
    }

    face_touch_dataset = {}
    face_touch_dataset['train'] = dataset.FaceTouchDataset(train_dict, data_transforms['train'], log_enabled=True)
    face_touch_dataset['val'] = dataset.FaceTouchDataset(val_dict, data_transforms['val'], log_enabled=True)
    face_touch_dataset['test'] = dataset.FaceTouchDataset(test_dict, data_transforms['val'], log_enabled=True)

    dataloaders = {}
    dataloaders['train'] = DataLoader(face_touch_dataset['train'], batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloaders['val'] = DataLoader(face_touch_dataset['val'], batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloaders['test'] = DataLoader(face_touch_dataset['test'], batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Let's initialize the device, in order to be able to train on GPU, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Initializing the model, the loss and the optimizer
    if args.select_squeezenet:
        net = model.SqueezeNet_fc(2)
    else:
        net = model.Resnet152_fc(2)

    net.to(device)
    criterion_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)# , momentum=args.momentum)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    # Setting the model path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Defining the training function
    def train_model(model, criterion, optimizer, num_epochs=500, resume_training=False, resuming_epoch=1, model_path=''):
        writer = SummaryWriter(args.logs_dir + args.experiment_name)

        # Let's manage situations in which we want to resume interrupted trainings
        if resume_training:
            if model_path == '':
                print('Please provide model_path to resume training or set resume_training=False')
            model_filename = model_path + str(resuming_epoch - 1) + '.pth'
            checkpoint = torch.load(model_filename)
            model.load_state_dict(checkpoint['state_dict'])

        # Let's track training time for better planning resource usage
        since = time.time()

        # We will choose the best model, defined as the model with the smallest validation loss
        min_val_loss = 100000.0

        for epoch in range(num_epochs):
            if resume_training:
                if epoch < resuming_epoch:
                    continue

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0

                # Iterate over data. Let's keep track of iteration and elapsed time
                iteration = 0
                t_0 = time.time()
                t_c = t_0

                for inputs, labels, filenames in dataloaders[phase]:
                    iteration = iteration + 1

                    # Visualize a batch of training data
                    # utils.show_batch(inputs, labels, img_mean, img_std)
                    # matplotlib.pyplot.show()

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Set to zero all the optimizer gradients
                    optimizer.zero_grad()

                    # Forward Pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        # Let's calculate the current loss and let's convert it in year error, for better understanding how the model is going
                        loss = criterion(outputs, labels)
                        # print('{} {}'.format(loss, loss))
                        # err = loss * train_date_std

                        # Backpropagate if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Increment the running loss for generating final statistics
                    running_loss += loss.item() * inputs.size(0)

                    # Periodically print training status information
                    if iteration % 100 == 0:
                        print("Exp {} -> {}: epoch: {: >4d} - {: >4d} out of {} ({: >2.2f}%) - loss: {:.4f} - cycle time: {:.4f} - elapsed time: {:.4f}".format(
                            selected_experiment, phase, epoch, iteration, len(dataloaders[phase]), 100 * iteration / len(dataloaders[phase]),
                            loss.item(), time.time() - t_c, time.time() - t_0))

                        t_c = time.time()

                # If in training, update the learning rate
                # if phase == 'train':
                #     scheduler.step()

                # Calculating epoch statistics and printing
                epoch_loss = running_loss / (args.batch_size*len(dataloaders[phase]))
                avg_err = epoch_loss # * train_date_std
                print('{} Loss: {:.4f} Avg Error: {:.4f}'.format(phase, epoch_loss, avg_err))

                # Saving the model on each epoch
                print('Saving..')
                model_filename = model_path + str(epoch) + '.pth'
                torch.save({'state_dict': model.state_dict()}, model_filename)
                print('..done!')

                writer.add_scalar(phase + '/loss', epoch_loss, epoch)
                writer.add_scalar(phase + '/avg_err', avg_err, epoch)

                # Saving the best model, based on validation accuracy
                if phase == 'val' and epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss

                    # model_filename = model_path + str(epoch) + '_best.pth'
                    # torch.save({'state_dict': model.state_dict()}, model_filename)

                    model_filename = model_path + 'best_model.pth'
                    torch.save({'state_dict': model.state_dict()}, model_filename)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # Load and return the trained model
        model_filename = model_path + 'best_model.pth'
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    train_model(net, criterion_loss, optimizer, num_epochs=args.num_epochs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--experiment_name', type=str, help='Experiment Name. It will be used for naming trained models and logs.', default='deployment')

    # output parameters
    parser.add_argument('--output_dir', type=str, help='Directory where the experiment results will be stored.', default='../models/')
    parser.add_argument('--logs_dir', type=str, help='Directory where the experiment logs will be stored by Tensorboard.', default='../runs/')

    # input parameters
    parser.add_argument('--csv_filepath', type=str, help='Path to the file listing the image filename, artistID, genre, style, date, title...', default='../data/all_data_info.csv')
    # parser.add_argument('--train_img_dir', type=str, help='Directory where the training images are located.', default='../data/train/')
    # parser.add_argument('--test_img_dir', type=str, help='Directory where the testing images are located.', default='../data/test/')

    # training parameters
    parser.add_argument('--select_squeezenet', help='Set to True if willing to use SqueezeNet instead of ResNet152.', type=utils.arg_str2bool, default=False)

    parser.add_argument('--batch_size', help='Batch Size.', type=int, default=16)
    parser.add_argument('--num_epochs', help='Number of training epochs.', type=int, default=500)
    parser.add_argument('--learning_rate', help='Learning Rate.', type=float, default=0.001)
    parser.add_argument('--momentum', help='Momentum.', type=float, default=0.9)
    parser.add_argument('--scheduler_step_size', help='Scheduler Step Size for adjusting learning rate.', type=int, default=10)
    parser.add_argument('--scheduler_gamma', help='Scheduler magnitude of learning rate adjustment.', type=float, default=0.1)

    parser.add_argument('--random_seed', help='Random Seed.', type=int, default=699)

    args = parser.parse_args()

    # print(args)
    main(args)
