from sdate.models.small_CNN import CNN_small
from sdate.datasets.tiff_tomogram_dataset import TIFFDataset

from argparse import ArgumentParser

import torch.nn as nn

from pytorch_base.experiment import PyTorchExperiment
from pytorch_base.base_loss import BaseLoss

import random

# from diffusers import UNet2DModel
import torchvision.models as models
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from torch.utils.data import Dataset

import torch


def top_k_accuracy(predicted_logits, reference, k=1):
    """
    Compute the top-k accuracy.

    Parameters:
    - predicted_logits: Tensor of shape (batch_size, num_classes), model predictions (logits).
    - reference: Tensor of shape (batch_size,), ground truth labels.
    - k: Number of top classes to consider for accuracy.

    Returns:
    - Accuracy as a float tensor.
    """
    # Get top-k predictions (returns values and indices, we need indices)
    with torch.no_grad():
        _, top_k_predictions = torch.topk(predicted_logits, k, dim=1)

        # Check if reference labels are in top-k predictions
        correct_predictions = top_k_predictions.eq(reference.view(-1, 1))  # Shape: (batch_size, k)

        # Compute accuracy
        return correct_predictions.any(dim=1).sum().float() / correct_predictions.size(0)


def get_dataset(dataset, dataset_expansion):
    torch.manual_seed(0)
    perm = torch.randperm(len(dataset))
    # we expand the train Set
    dataset_expansion = max(1, dataset_expansion)  # At least 1
    trainSet = torch.utils.data.Subset(dataset, torch.cat([torch.arange(len(dataset)) for _ in range(dataset_expansion)]))
    testSet = torch.utils.data.Subset(dataset, perm[round(0.99 * len(dataset)):])
    return dataset, trainSet, testSet


def get_resnet(diff_step_bins, pretrained):
    if pretrained:
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, diff_step_bins)
    else:
        model = models.resnet18(pretrained=False, num_classes=diff_step_bins)
    weights = model.conv1.weight.mean(dim=1).unsqueeze(1).clone()
    model.conv1 = nn.Conv2d(in_channels=1,  # Change from 3 to 1 channel
                            out_channels=64,  # Keep the original output channels
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
    model.conv1.weight = nn.Parameter(weights)

    return model


class prediction_loss(BaseLoss):

    def __init__(self, device, noise_scheduler, args):
        stats_names = ["loss", "accuracy", "accuracy5", "accuracy10"]
        super(prediction_loss, self).__init__(stats_names)
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.args = args

    def compute_loss(self, instance, model):

        ce = torch.nn.CrossEntropyLoss()
        img = instance
        img = img.to(self.device)

        bs = len(img)

        # We train only in timesteps that will be used for inference
        all_timesteps = torch.linspace(
            self.noise_scheduler.timesteps.max(),
            self.noise_scheduler.timesteps.min(),
            self.args["diff_step_bins"]
        ).int()

        timestep_indices = torch.randint(
            0, len(all_timesteps), (bs,)
        )
        timesteps = all_timesteps[timestep_indices].long()
        timestep_indices = timestep_indices.to(self.device)

        noise = torch.randn_like(img)
        x_t = self.noise_scheduler.add_noise(img, noise, timesteps)

        model.zero_grad()
        pred_timesteps = model(x_t)

        loss = ce(pred_timesteps, timestep_indices)
        acc = top_k_accuracy(pred_timesteps, timestep_indices, k=1)
        acc5 = top_k_accuracy(pred_timesteps, timestep_indices, k=5)
        acc10 = top_k_accuracy(pred_timesteps, timestep_indices, k=10)

        return loss, {"loss": loss, "accuracy": acc, "accuracy5": acc5, "accuracy10": acc10}


def load_model(model, optimizer, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"model loaded from checkpoint {model_path}")
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"optimizer loaded from checkpoint {model_path}")


def run_experiment(
    args,
    model_instance,
    model_path,
    verbose=True,
    dataset_expansion=1
):
    """
    Run a PyTorch experiment with the provided parameters.

    Parameters correspond to those defined in your original argparse.
    """
    import torch
    import random

    # Prepare dataset kwargs (assumes TIFFDataset and get_dataset are defined)
    kwargs = {
        "path": args['dataset_path'],
        "im_size": args['im_size'],
        "train_transform": False,
        "rescale": args['rescale'],
        "normalize_range": False,
    }
    original_dataset, trainSet, testSet = get_dataset(TIFFDataset(**kwargs), dataset_expansion)
    if dataset_expansion  == 0:
        return original_dataset

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device:", device)
    model_instance = model_instance.to(device)

    # Choose the optimizer based on the provided option.
    if args['optimizer'] == "SGD":
        optimizer_instance = torch.optim.SGD(model_instance.parameters(), lr=args['learning_rate'], momentum=0.9)
    else:
        optimizer_instance = torch.optim.AdamW(model_instance.parameters(), lr=args['learning_rate'])

    model_instance.train()
    # Optionally load a checkpoint.
    if args['load_checkpoint'] != "":
        try:
            load_model(model_instance, optimizer_instance, args['load_checkpoint'])
        except Exception as e:
            print("Checkpoint not found, initializing model randomly.")

    # Create the experiment (assumes PyTorchExperiment and prediction_loss are defined)
    exp = PyTorchExperiment(
        train_dataset=trainSet,
        test_dataset=testSet,
        batch_size=args['batch_size'],
        model=model_instance,
        loss_fn=prediction_loss(device, DDPMScheduler(num_train_timesteps=1000), args),
        checkpoint_path=model_path,
        experiment_name=args['exp_name'],
        with_wandb=args['wandb'],
        num_workers=torch.get_num_threads() if torch.cuda.is_available() else 0,
        seed=args['seed'],
        args=args,
        save_always=True,
        verbose=verbose,
        is_logging_epoch=args['is_logging_epoch'] if 'is_logging_epoch' in args else lambda x:True
    )

    # Set up the learning rate scheduler (assumes get_cosine_schedule_with_warmup is defined)
    lr_scheduler_instance = get_cosine_schedule_with_warmup(
        optimizer=optimizer_instance,
        num_warmup_steps=50,
        num_training_steps=(len(trainSet) * args['epochs']),
    )

    # Start training.
    exp.train(args['epochs'], optimizer_instance, milestones=[10000], gamma=0.1, scheduler=lr_scheduler_instance)
    return original_dataset

def get_model(model_type, args, device):
    if model_type == "small_CNN":
        model = CNN_small(args['diff_step_bins'])
    else:
        model = get_resnet(diff_step_bins=args['diff_step_bins'], pretrained=args['pretrained'])

    if args['train_head_only']:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    return model.to(device)

if __name__ == '__main__':
    # import lovely_tensors as lt
    #
    # lt.monkey_patch()

    parser = ArgumentParser(description="PyTorch experiments")
    parser.add_argument("--batch_size", default=50, type=int, help="batch size of every process")
    parser.add_argument("--epochs", default=50, type=int, help="number of epochs to train")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--exp_name", default='random_experiment', type=str, help="Experiment name")
    parser.add_argument('--wandb', action='store_true', help="Use wandb")
    parser.add_argument("--load_checkpoint", default='', type=str, help="name of models in folder checkpoints to load")
    parser.add_argument("--seed", default=-1, type=int, help="Random seed")
    parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam"], type=str, help="Optimizer to be used")

    parser.add_argument("--model", default="small_CNN", choices=["small_CNN", "Resnet"], type=str, help="Optimizer to be used")
    parser.add_argument("--diff_step_bins", default=50, type=int, help="Number of diffusion steps")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset file or folder")
    parser.add_argument("--im_size", type=int, default=512,
                        help="In the case of tiff, the size of the crops to split the tiff files")
    parser.add_argument("--rescale", type=int, default=512, help="The side length of the images in the dataset")
    parser.add_argument('--train_head_only', action='store_true', help="Use wandb")
    parser.add_argument('--pretrained', action='store_true', help="Load pretrained model")

    args = vars(parser.parse_args())
    args["seed"] = random.randint(0, 20000) if args["seed"] == -1 else args["seed"]


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_path = f"checkpoints/{args['exp_name']}.pt"

    model_instance = get_model(args['model'], args, device) # Get the model

    run_experiment(args, model_instance, model_path, verbose=True, dataset_expansion=10)


