import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from pytorch_base.stats_tracker import StatsTracker
from sdate.losses.Base_loss import Base_Loss
from sdate.training.train_step_prediction import top_k_accuracy
import wandb

# Function to compute entropy
def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # Avoid log(0)
    return entropy

class DTE_Loss(Base_Loss):
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    all_timesteps = torch.linspace(
        noise_scheduler.timesteps.max(),
        noise_scheduler.timesteps.min(),
        50
    ).int()


    @staticmethod
    def compute_train_loss(model, img, device):
        ce = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            img = img.clone().to(device)
            img = DTE_Loss.process_data(img)
            all_timesteps = DTE_Loss.all_timesteps

            bs = len(img)
            timestep_indices = torch.randint(
                0, len(all_timesteps), (bs,)
            )
            # one quarter of the indices deal with the test case
            timestep_indices[:1] = all_timesteps[-1]

            timesteps = all_timesteps[timestep_indices].long()
            timestep_indices = timestep_indices.to(device)

            noise = torch.randn_like(img)
            x_t = DTE_Loss.noise_scheduler.add_noise(img, noise, timesteps)

        pred_timesteps = model(x_t.to(device))

        loss = ce(pred_timesteps, timestep_indices)
        acc = top_k_accuracy(pred_timesteps, timestep_indices, k=1)
        return loss, acc

    @staticmethod
    def compute_test_result(args, data_loader, model, device, index_i, filename):
        all_timesteps = DTE_Loss.all_timesteps
        tracker = StatsTracker("loss_acc", ["loss", "acc", "entropy", "argmax"])

        with torch.no_grad():
            for img in data_loader:
                loss, acc, pred_timesteps = DTE_Loss.compute_test_loss(args, model, img, device)
                tracker.add({"loss": loss.item(), "acc": acc.item()}, len(img))

                # Compute entropy per sample
                entropy = compute_entropy(pred_timesteps)  # Shape: [batch_size]
                avg_entropy = torch.mean(entropy).item()
                tracker.add({"entropy": avg_entropy}, len(img))

                # Compute argmax per sample
                arg_max = torch.argmax(pred_timesteps, dim=-1)  # Shape: [batch_size]
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

    @staticmethod
    def compute_test_loss(args, model, img, device):
        all_timesteps = DTE_Loss.all_timesteps
        ce = torch.nn.CrossEntropyLoss()
        img = img.to(device)
        img = DTE_Loss.process_data(img)
        bs = len(img)

        timestep_idx = args['test_timestep'] # Use the specified timestep for evaluation
        timestep_indices = torch.tensor([timestep_idx]).repeat(bs).to(device)

        timesteps = all_timesteps[timestep_indices.cpu()].long()
        noise = torch.randn_like(img)
        x_t = DTE_Loss.noise_scheduler.add_noise(img, noise, timesteps)
        pred_timesteps = model(x_t.to(device))

        loss = ce(pred_timesteps, timestep_indices)  # Shape: [batch_size]
        acc = top_k_accuracy(pred_timesteps, timestep_indices, k=1)
        return loss, acc, pred_timesteps