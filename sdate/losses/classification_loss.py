import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from pytorch_base.stats_tracker import StatsTracker
from sdate.losses.Base_loss import Base_Loss
from sdate.training.train_step_prediction import top_k_accuracy
import wandb


class Classification_Loss(Base_Loss):
    def __init__(self):
        super.__init__()

    @staticmethod
    def compute_test_loss(args, model, img, device):
        return Rotation_Loss.compute_loss(model, img, device, random=False)

    @staticmethod
    def compute_loss(model, img, device, random=True):

        logloss = torch.nn.BCEWithLogitsLoss()
        img = img.to(device)
        pred = model(img)

        loss = logloss(pred_angles, angle_indices)
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