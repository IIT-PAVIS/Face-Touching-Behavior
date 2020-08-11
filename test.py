import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import argparse
import time
import numpy as np
import pickle
import pandas as pd

import model
import utils
import dataset

def main(args):
    ## Initializing the model, the loss and the optimizer
    if args.select_squeezenet:
        net = model.SqueezeNet_fc(2)
    else:
        net = model.Resnet152_fc(2)

    # Loading the model weights
    checkpoint = torch.load(args.model_filename)
    net.load_state_dict(checkpoint['state_dict'])

    # Instantiating the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Defining image transformation
    image_size = 300
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(img_mean, img_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(img_mean, img_std)
        ]),
    }

    experiments = pickle.load(open(args.json_path, "rb"))

    # Testing Experiments[selected_experiment]
    experiment = experiments[args.selected_experiment]

    ## Initializng pyTorch dataloaders
    test_dict = {
        "imgs": experiment['test_imgs'],
        "labels": experiment['test_labels']
    }

    face_touch_dataset = {}
    face_touch_dataset['test'] = dataset.FaceTouchDataset(test_dict, data_transforms['val'], log_enabled=True)

    dataloaders = {}
    dataloaders['test'] = DataLoader(face_touch_dataset['test'], batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Each epoch has a training and validation phase
    results_filenames = []
    results_predictions = []
    results_labels = []

    for phase in ['test']:
        if phase == 'train':
            net.train()
        else:
            net.eval()

        # Iterate over data. Let's keep track of iteration and elapsed time
        iteration = 0
        t_0 = time.time()
        t_c = t_0

        for inputs, labels, filenames in dataloaders[phase]:
            # if iteration>10:
            #     break

            iteration = iteration + 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            _, predictions = torch.max(outputs, 1)

            # Periodically print training status information
            if iteration % 100 == 0:
                print(
                    "Exp {} -> {}: epoch: {: >4d} - {: >4d} out of {} ({: >2.2f}%) - loss: {:.4f} - cycle time: {:.4f} - elapsed time: {:.4f}".format(
                        args.selected_experiment, phase, 0, iteration, len(dataloaders[phase]),
                        100 * iteration / len(dataloaders[phase]),
                        0, time.time() - t_c, time.time() - t_0))

                t_c = time.time()

            # Visualize a batch of training data
            # if iteration < 50:
            #     utils.show_batch(inputs, predictions, img_mean, img_std)
            #     matplotlib.pyplot.show()

            # Save input/output pair
            results_filenames.extend(filenames)
            results_predictions.extend(predictions.cpu().numpy().tolist())
            results_labels.extend(labels.cpu().numpy().tolist())

    results_filenames_np = np.asarray(results_filenames)
    results_predictions_np = np.asarray(results_predictions)
    results_labels_np = np.asarray(results_labels)
    results = np.concatenate((np.expand_dims(results_filenames_np, axis=1), np.expand_dims(results_predictions_np, axis=1)), axis=1)
    results = np.concatenate((results, np.expand_dims(results_labels_np, axis=1)), axis=1)
    results = np.concatenate((results, np.expand_dims(results_labels_np == results_predictions_np, axis=1)), axis=1)
    # print(results)

    # Saving the results to CSV
    print('Saving the results to {}...'.format(args.results_out_path))
    pd.DataFrame(results).to_csv(args.results_out_path)
    print('Done!'.format(args.results_out_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_filename', help='Path to the file with the model weights.', type=str, default='/home/mbustreo/Projects/gitLab/face-touch/src/best_model_3.pth')
    parser.add_argument('--json_path', help='Path to the file with the json file with the splits.', type=str, default='/home/mbustreo/Projects/gitLab/face-touch/data/experiments.json')
    parser.add_argument('--selected_experiment', help='Number of the selected experiment.', type=int, default=3)

    parser.add_argument('--results_out_path', help='Path to the file where saving the results', type=str, default='/home/mbustreo/Projects/gitLab/face-touch/data/results_3.csv')

    parser.add_argument('--batch_size', help='Batch Size.', type=int, default=12)
    parser.add_argument('--select_squeezenet', help='Set to True if willing to use SqueezeNet instead of ResNet152.', type=utils.arg_str2bool, default=False)

    args = parser.parse_args()

    main(args)
