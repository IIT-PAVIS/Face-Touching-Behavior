import matplotlib.pyplot as plt
import numpy as np
import re
import math
import time
import torch
from PIL import Image, ImageFile

# Date selection functions
def getSample(df, percentage_too_keep = 0.01):
    return df.sample(withReplacement=False, fraction=percentage_too_keep)

# Visualizing functions
def plotHistogram(pandas_df):
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.rcParams.update({'font.size': 22})

    pandas_df.plot(kind='barh', x='date', y='num_of_entries')
    plt.show()

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.pause(0.001)  # pause a bit so that plots are updated

def show_batch(image_batch, label_batch, img_mean, img_std, label_mean, label_std):
    batch_size = len(image_batch)
    num_x_imgs = int(math.sqrt(batch_size))
    num_y_imgs = int(len(image_batch) / num_x_imgs) + 1

    label_batch = label_batch * label_std + label_mean
    plt.figure(figsize=(15, 15))
    for n in range(batch_size):
        ax = plt.subplot(num_x_imgs, num_y_imgs, n + 1)
        image = image_batch[n].permute(1, 2, 0).numpy()
        image = img_std * image + img_mean

        plt.imshow(image.clip(0, 1))
        plt.title(str(int(label_batch[n].item())))
        plt.axis('off')


def show_batch_numpy(image_batch, label_batch):
    batch_size = len(image_batch)
    num_x_imgs = int(math.sqrt(batch_size))
    num_y_imgs = int(len(image_batch) / num_x_imgs) + 1

    label_batch = label_batch
    plt.figure(figsize=(15, 15))
    for n in range(batch_size):
        ax = plt.subplot(num_x_imgs, num_y_imgs, n + 1)
        image = Image.open(image_batch[n])
        image = np.array(image)

        plt.imshow(image)
        plt.title(str(int(label_batch[n].item())))
        plt.axis('off')


def show_batch(image_batch, label_batch, img_mean, img_std):
    batch_size = len(image_batch)
    num_x_imgs = int(math.sqrt(batch_size))
    num_y_imgs = int(len(image_batch) / num_x_imgs) + 1

    label_batch = label_batch
    plt.figure(figsize=(15, 15))
    for n in range(batch_size):
        ax = plt.subplot(num_x_imgs, num_y_imgs, n + 1)
        image = image_batch[n].permute(1, 2, 0).cpu().numpy()
        image = img_std * image + img_mean

        plt.imshow(image.clip(0, 1))
        plt.title(str(int(label_batch[n].item())))
        plt.axis('off')

# We define a function for visualizing inputs, date estimation and ground truth.
def visualize_model_results(model, device, dataloader, img_mean, img_std, label_mean, label_std, num_images=10):
    model.eval()

    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()

    inputs, labels, filenames = next(iter(dataloader))

    inputs = inputs.to(device)
    preds = model(inputs) * label_std + label_mean
    labels = labels.float().view(-1, 1) * label_std + label_mean

    # Defining number of images per row and per column
    num_images = min(num_images, len(inputs))

    num_x_imgs = int(math.sqrt(num_images))
    num_y_imgs = int(num_images / num_x_imgs) + 1

    plt.figure(figsize=(15, 15))
    for n in range(num_images):
        ax = plt.subplot(num_x_imgs, num_y_imgs, n + 1)
        image = inputs[n].permute(1, 2, 0).cpu().numpy()
        image = img_std * image + img_mean

        plt.imshow(image.clip(0, 1))
        ax.axis('off')
        ax.set_title('label: {} \npredicted: {} \nerror: {} years'.format(math.ceil(labels[n].item()),
                              math.ceil(preds[n].item()), math.ceil(preds[n].item() - labels[n].item())))

# Date Cleansing functions
def keep_date_only(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(int(float(x)))
        except ValueError:
            kdo = re.sub("[^0-9]", "", x)
            if len(kdo)>0:
                return float(kdo)

# Parameter parsing functions
def arg_str2bool(v):
    import argparse

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Model evaluation functions
def evaluate_model(model, device, dataloader, label_mean, label_std):
    labels_all = []
    preds_all = []
    filenames_all = []

    iteration = 0
    t_0 = time.time()
    t_c = t_0

    model.eval()
    with torch.no_grad():
        for inputs, labels, filenames in dataloader:
            iteration = iteration + 1

            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.float().view(-1, 1) * label_std + label_mean

            preds = model(inputs) * label_std + label_mean

            labels_all.append(labels)
            preds_all.append(preds)
            filenames_all.append(filenames)

            # Periodically print training status information
            if iteration % 10 == 0:
                print("Iteration {: >4d} out of {} ({: >2.2f}%) - cycle time: {:.4f} - elapsed time: {:.4f}".format(
                    iteration, len(dataloader), 100 * iteration / len(dataloader), time.time() - t_c,
                                                time.time() - t_0))

    return filenames_all, labels_all, preds_all

# Conversion functions
def tensor_num_vec_to_numpy(vector):
    for i in range(len(vector)):
        out_i = vector[i].cpu().detach().numpy()
        if i>0:
            out = np.concatenate((out, out_i))
        else:
            out = out_i

    return out

def tensor_str_vec_to_numpy(vector):
    for i in range(len(vector)):
        out_i = vector[i]
        if i>0:
            out = np.concatenate((out, out_i))
        else:
            out = out_i

    return out