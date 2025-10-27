import gc
import os
import copy
import matplotlib.pyplot as plt
# import lovely_tensors as lt
# lt.monkey_patch()
import wandb
from torch.optim.lr_scheduler import MultiStepLR
from tqdm.auto import tqdm
import glob
import subprocess
from pathlib import Path
from argparse import ArgumentParser
import random

from sdate.datasets.tiff_dataset import TiffDataset
from sdate.datasets.tiff_tomogram_dataset import TIFFDataset
from sdate.models.small_CNN import CNN_small
from sdate.training.train_step_prediction import get_resnet, top_k_accuracy
from pytorch_base.stats_tracker import StatsTracker
from sdate.models.resnet_grayscale import ResnetGrayscale



import torch

def compute_dataset(args, dataset_path, clip_range=None):
    # kwargs = {
    #     "path": dataset_path,
    #     "im_size": args['im_size'],
    #     "train_transform": False,
    #     "rescale": args['rescale'],
    #     "normalize_range": False,
    # }
    # torch.manual_seed(0)
    # dataset = TIFFDataset(**kwargs)
    dataset = TiffDataset(dataset_path, rescale=args['rescale'], clip_range=clip_range)
    return dataset

def get_model(model_type, args, device):
    if model_type == "small_CNN":
        model = CNN_small(args['diff_step_bins'])
    else:
        # model = get_resnet(diff_step_bins=args['diff_step_bins'], pretrained=args['pretrained'])
        model = ResnetGrayscale(pretrained=args['pretrained'], out_features=args['diff_step_bins'], only_fc=args['train_head_only'])

    # if args['train_head_only']:
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     for param in model.fc.parameters():
    #         param.requires_grad = True

    if args['load_checkpoint'] != "":
        try:
            load_model_only(model, args['load_checkpoint'])
        except Exception as e:
            print("Checkpoint not found, initializing model randomly.")
    return model.to(device)

def list_subfolders(directory):
    for firstFile in os.scandir(directory):
        break
    if os.path.isdir(firstFile):
        return [f.path for f in os.scandir(directory) if f.is_dir()]
    else:
        image_paths = sorted(
            glob.glob(os.path.join(directory, '*.tif')) +
            glob.glob(os.path.join(directory, '*.tiff'))
        )
        return image_paths

def load_model_only(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"model loaded from checkpoint {model_path}")

# Function to compute results and log both individual and aggregated values

def test_model(test_bs, stabilized, anomaly_detected, args, model, currentSet, device, i, filename):
    # We start by testing the current model
    data_loader = torch.utils.data.DataLoader(currentSet, batch_size=test_bs, shuffle=False, num_workers=0)
    model.eval()
    result = LossFunction.compute_test_result(args, data_loader, model, device, i, filename)

    avg_argmax = result["avg_loss"]
    avg_acc = result["avg_acc"]

    # Append the value (convert to float if necessary)
    recent_values.append(avg_argmax.item() if isinstance(avg_argmax, torch.Tensor) else avg_argmax)
    # Convert the latest window to a tensor
    window = torch.tensor(recent_values[-window_size:])
    # Compute standard deviation over the window
    if torch.std(window) < threshold_std:
        if not args['wandb']:
            print(f"Stabilized at iteration {filename} {torch.std(window)}")
        # Trigger your event here, e.g., break out of the loop or call another function.
        stabilized = True
    elif torch.std(window) > 10 * threshold_std:
        # Trigger anomaly detection here
        if not args['wandb']:
            print(f"Anomaly detected {filename} {torch.std(window)}")
        anomaly_detected = True
        stabilized = False

    torch.cuda.empty_cache()
    del currentSet, data_loader
    return stabilized, anomaly_detected, avg_acc



def train(args, model, currentSet, device, i, filename):
    file_number = int(filename[-5:])
    data_loader = torch.utils.data.DataLoader(currentSet, batch_size=args['batch_size'], shuffle=True, num_workers=0)
    if args['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9)
        scheduler = MultiStepLR(optimizer, milestones=args['scheduler'], gamma=0.1)
        print("Training with SGD")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
        print("Training with Adam")
        scheduler = None

    min_test_loss = float('inf')
    # initialize min_model_state_dict with a state dict of the model, but all with zeros
    min_model_state_dict = None

    for epoch in range(args['epochs']):
        model.train()
        train_tracker = StatsTracker("Train", {"loss", "acc"})

        for img in data_loader:
            optimizer.zero_grad()
            bs = len(img)
            loss, acc = LossFunction.compute_train_loss(model, img, device)

            train_tracker.add({"loss": loss.item(), "acc": acc.item()}, bs)

            loss.backward()
            optimizer.step()


        # if the target accuracy is reached, break the loop
        if train_tracker.get_mean("acc") > args['target_acc']:
            print(f"Target accuracy reached at epoch {epoch}")
            break
        if scheduler is not None:
            scheduler.step()

        data_loader = torch.utils.data.DataLoader(currentSet, batch_size=2*args['batch_size'], shuffle=False)
        with torch.no_grad():
            test_tracker = StatsTracker("Test", {"loss", "acc"})
            model.eval()
            for img in data_loader:
                bs = len(img)
                loss, acc = LossFunction.compute_train_loss(model, img, device)
                test_tracker.add(
                    {"loss": loss.item(), "acc": acc.item()}, bs)

        # If we see the min test_loss so far, then we store the model
        if test_tracker.get_mean("loss") < min_test_loss:
            min_test_loss = test_tracker.get_mean("loss")
            min_model_state_dict = copy.deepcopy(model.state_dict())



        # Log the loss and accuracy to wandb
        if args['wandb']:
            wandb.log({
                f"train_epoch": epoch,
                f"train_loss_{file_number}": train_tracker.get_mean("loss"),
                f"initial_test_loss_{file_number}": test_tracker.get_mean("loss"),
                f"initial_test_acc_{file_number}": test_tracker.get_mean("acc"),
                f"train_accuracy_{file_number}": train_tracker.get_mean("acc"),
                "filename": filename,
                "file_number": file_number
            })
        else:
            print(
                f"Epoch {epoch} - Train Loss: {train_tracker.get_mean('loss')}, "
                f"Test Loss: {test_tracker.get_mean('loss')}, "
                f"Train Acc: {train_tracker.get_mean('acc')}, "
                f"Test Acc: {test_tracker.get_mean('acc')}"
            )

    # Recover the model with the lowest test loss and map model to device
    model.load_state_dict(min_model_state_dict)

    del data_loader, train_tracker, optimizer, min_model_state_dict

    if args["save_model"]:
        torch.save(model.state_dict(), f"checkpoints/AnomalyStepPred_{filename}.pt")
        print(f"Model saved at checkpoints/AnomalyStepPred_{filename}.pt")

# if this is main
if __name__ == '__main__':
    parser = ArgumentParser(description="PyTorch experiments")
    parser.add_argument("--diff_step_bins", default=50, type=int, help="Num od diffusion timesteps to use")
    parser.add_argument('--pretrained', action='store_true', help="Use pretrained models")
    parser.add_argument('--train_head_only', action='store_true', help="Train only the head of the model")
    parser.add_argument('--save_model', action='store_true', help="Save the model after training")

    parser.add_argument('--reflect', action='store_true', help="Reflect the image values by multiplying them with -1")

    parser.add_argument("--batch_size", default=50, type=int, help="batch size of every process")
    parser.add_argument("--epochs", default=50, type=int, help="number of epochs to train")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument("--threshold_std", default=0.05, type=float, help="Threshold for stabilization")

    parser.add_argument("--target_acc", default=1.0, type=float, help="Target accuracy to stop training loop when reached")
    parser.add_argument("--test_timestep", default=49, type=int, help="Target timestep to test the model")
    parser.add_argument('--use_acc_restart', action='store_true', help="Use accuracy as a train restart criteria")

    parser.add_argument("--loss_function", default="DTE", type=str, choices=["DTE", "Rotation"], help="Loss function to be used")
    parser.add_argument(
        "--data_path",
        default="/das/work/p22/p22274/Cordierite/downsampled_cordierite" if torch.cuda.is_available() else "/Users/lfbarba/GitHub/deep-anomaly-detection/data/downsampled_cordierite/",
        type=str, help="Path with data"
    )

    parser.add_argument(
        "--pretrain_path", default="", type=str, help="Path to data to pretrain data"
    )

    parser.add_argument('--wandb', action='store_true', help="Use wandb")
    parser.add_argument("--load_checkpoint", default='', type=str, help="name of models in folder checkpoints to load")
    parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam"], type=str, help="Optimizer to be used")
    parser.add_argument("--im_size", type=int, default=512, help="In the case of tiff, the size of the crops to split the tiff files")
    parser.add_argument("--rescale", type=int, default=-1, help="The side length of the images in the dataset")
    parser.add_argument("--seed", default=-1, type=int, help="Random seed")
    parser.add_argument("--scheduler", default="[500000000]", type=str, help="scheduler decrease after epochs given")
    parser.add_argument("--clip_range", type=str, default="", help="The side length of the images in the dataset")

    args = vars(parser.parse_args())
    temp = args["scheduler"].replace(" ", "").replace("[", "").replace("]", "").split(",")
    args["scheduler"] = [int(x) for x in temp]
    args["seed"] = random.randint(0, 20000) if args["seed"] == -1 else args["seed"]
    if args["clip_range"] == "":
        args["clip_range"] = None
    else:
        temp = args["clip_range"].replace(" ", "").replace("[", "").replace("]", "").split(",")
        args["clip_range"] = [int(x) for x in temp]

    if args['loss_function'] == "DTE":
        from sdate.losses.dte_loss import DTE_Loss as LossFunction
    elif args['loss_function'] == "Rotation":
        from sdate.losses.rotation_loss import Rotation_Loss as LossFunction

    # Get all subfolders
    directory_path = args['data_path']
    subfolders = list_subfolders(directory_path)
    subfolders = [x for x in sorted(subfolders)]
    print("subfolders", len(subfolders))

    if args['wandb']:
        # Log in to wandb
        wandb_api_key = "d6f99b98acf9c1a284aa2ba5830f3eca60fde2f0"
        wandb.login(key=wandb_api_key)  # Log in to wandb

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
    results = {}

    for model_type, test_bs in zip(["Resnet"], [250, 250]):
        model_path = f"checkpoints/AnomalyStepPred_{model_type}.pt"
        print(f"training {model_type} model")
        results[model_type] = []

        model = get_model(model_type, args, device)

        run_id = random.randint(0, 10000)

        epochs = args['epochs']

        # Start a new wandb run for this model type
        if args['wandb']:
            wandb.init(
                project=f"SDATE_DTE_NEW_{model_type}",
                name=f"{str(Path(Path(args['data_path']).parent).stem)[:7]}_{run_id}",
                config={"model_type": model_type, "run_id": run_id, "data_path":args["data_path"], "pretrain_path":args["pretrain_path"]}
            )

        # Parameters to tune
        window_size = 4  # number of recent iterations to check
        threshold_std = args['threshold_std']  # threshold for standard deviation to consider "stable"
        stabilized = False # flag to update the model
        anomaly_detected = False # flag to detect anomaly
        avg_acc = 0
        # Buffer to store recent avg_argmax values
        recent_values = []

        if args["pretrain_path"] == "":
            args["pretrain_path"] = subfolders[0]
        currentSet = compute_dataset(args, args["pretrain_path"], clip_range=args['clip_range'])
        # compute mean and std from entire dataset
        mean_tracker = StatsTracker("Mean", {"mean"})
        std_tracker = StatsTracker("Std", {"std"})
        for img in currentSet:
            mean_tracker.add({"mean": img.mean().item()}, 1)
            std_tracker.add({"std": img.std().item()}, 1)
        LossFunction.mean = mean_tracker.get_mean("mean");
        LossFunction.std = std_tracker.get_mean("std");
        LossFunction.reflect = args['reflect']


        print(f"Pretraing on {args['pretrain_path']}")
        train(
            args, model, currentSet, device, 0,
            Path(args["pretrain_path"]).stem
        )
        del currentSet
        torch.cuda.empty_cache()  # Clears GPU cache explicitly
        gc.collect()  # Frees Python objects from CPU memory

        # use tqdm as the external loop, there will be internal tqdm loops
        for i, dataset_path in tqdm(enumerate(subfolders), desc="Folder Loop", total=len(subfolders)):
            filename = Path(dataset_path).stem

            currentSet = compute_dataset(args, dataset_path, clip_range=args['clip_range'])
            stabilized, anomaly_detected, avg_acc = test_model(
                test_bs,
                stabilized, anomaly_detected,
                args, model, currentSet, device, i, filename
            )
            del currentSet

            doTraining = (
                not args['use_acc_restart'] and anomaly_detected and stabilized
            )  or (
                args['use_acc_restart'] and avg_acc == 0
            )
            if doTraining:
                print("Doing training due to doTraining==True")
                # If we have detected an anomaly and the model has stabilized afterward
                anomaly_detected = False
                stabilized = False
                # Train the model with the new dataset
                currentSet = compute_dataset(args, dataset_path, clip_range=args['clip_range'])
                train(args, model, currentSet, device, i, filename)
                recent_values = [] # Reset the buffer
                del currentSet

            torch.cuda.empty_cache()  # Clears GPU cache explicitly
            gc.collect()  # Frees Python objects from CPU memory


        del model
        if wandb.run:
            wandb.finish()