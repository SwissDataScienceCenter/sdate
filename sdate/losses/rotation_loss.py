import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from pytorch_base.stats_tracker import StatsTracker
from sdate.losses.Base_loss import Base_Loss
from sdate.training.train_step_prediction import top_k_accuracy
import wandb

# Function to compute entropy
def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # Avoid log(0)
    return entropy

def rotate_batch(images, angles, mode='bilinear', padding_mode='zeros'):
    """
    images: Tensor of shape (B, C, H, W)
    angles: Tensor of shape (B,) with angles in degrees
    """
    B, C, H, W = images.shape
    angles_rad = torch.deg2rad(angles)

    # Compute affine rotation matrices
    theta = torch.zeros(B, 2, 3, device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = torch.cos(angles_rad)
    theta[:, 0, 1] = -torch.sin(angles_rad)
    theta[:, 1, 0] = torch.sin(angles_rad)
    theta[:, 1, 1] = torch.cos(angles_rad)

    # Generate grids
    grid = F.affine_grid(theta, images.size(), align_corners=False)

    # Sample from grid
    rotated_images = F.grid_sample(images, grid, mode=mode, padding_mode=padding_mode, align_corners=False)

    return rotated_images

class Rotation_Loss(Base_Loss):
    all_angles = torch.linspace(-90, 90, 50)
    def __init__(self):
        super.__init__()

    @staticmethod
    def compute_test_loss(args, model, img, device):
        return Rotation_Loss.compute_loss(model, img, device, random=False)

    @staticmethod
    def compute_loss(model, img, device, random=True):
        all_angles = Rotation_Loss.all_angles
        ce = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            img = img.to(device)
            img = Rotation_Loss.process_data(img)
            bs = len(img)
            # rotate image by an arbitrary degree between -90 and 90 degrees using torch build in functions
            if random:
                angle_indices = torch.randint(0, len(all_angles) - 1, (bs,))
            else:
                angle_indices = torch.tensor([len(all_angles)-1] * bs)

            img = rotate_batch(img, all_angles[angle_indices])
            angle_indices = angle_indices.to(device)

        pred_angles = model(img)

        loss = ce(pred_angles, angle_indices)
        acc = top_k_accuracy(pred_angles, angle_indices, k=1)
        return loss, acc, pred_angles

    @staticmethod
    def compute_train_loss(model, img, device):
        loss, acc, _ =  Rotation_Loss.compute_loss(model, img, device, random=True)
        return loss, acc


    @staticmethod
    def compute_test_result(args, data_loader, model, device, index_i, filename):
        all_angles = Rotation_Loss.all_angles
        tracker = StatsTracker("loss_acc", ["loss", "acc", "entropy", "argmax"])

        with torch.no_grad():
            for img in data_loader:
                loss, acc, pred_angles = Rotation_Loss.compute_test_loss(args, model, img, device)
                tracker.add({"loss": loss.item(), "acc": acc.item()}, len(img))

                # Compute entropy per sample
                entropy = compute_entropy(pred_angles)  # Shape: [batch_size]
                avg_entropy = torch.mean(entropy).item()
                tracker.add({"entropy": avg_entropy}, len(img))

                # Compute argmax per sample
                arg_max = torch.argmax(pred_angles, dim=-1)  # Shape: [batch_size]
                avg_argmax = torch.mean(arg_max.float()).item()
                tracker.add({"argmax": avg_argmax}, len(img))

        # Compute averages
        avg_loss = tracker.get_mean("loss")
        avg_entropy = tracker.get_mean("entropy")
        avg_argmax = tracker.get_mean("argmax")
        avg_acc = tracker.get_mean("acc")

        del tracker

        # Log aggregated values to wandb
        if args['wandb'] and wandb.run:
            wandb.log({
                "index": index_i,
                "avg_loss": avg_loss,
                "avg_entropy": avg_entropy,
                "avg_acc": avg_acc,
                "avg_argmax": avg_argmax,
                "file_number": int(filename[-5:]),
                "level": index_i,
                "filename": filename
            })
        else:
            print({
                "avg_loss": avg_loss,
                "avg_entropy": avg_entropy,
                "avg_argmax": avg_argmax,
                "avg_acc": avg_acc
            })

        return {
            "avg_loss": avg_loss,
            "avg_entropy": avg_entropy,
            "avg_argmax": avg_argmax,
            "avg_acc": avg_acc
        }