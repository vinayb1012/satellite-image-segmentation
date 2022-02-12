import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as td
from torch import nn
from tqdm import tqdm

from eval_utils import *
# Taken from https://github.com/milesial/Pytorch-UNet
from model.unet_model import UNet


def add_arguments(parser):
    parser.add_argument("--session-name", default="runs", type=str,
                        help="Session name to save model snapshots and outputs")
    parser.add_argument("--channels",
                        help="Use only RGB or all channels for training",
                        default="rgb",
                        choices=["rgb", "multispectral"])
    parser.add_argument("--model", type=str, help="Checkpoint to resume from")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train")
    parser.add_argument("--batch-size", default=20, type=int, help="Batch size for training")
    parser.add_argument("--learning-rate", default=0.0001, type=float)
    parser.add_argument("--eval-freq", default=1, type=int, help="Evaluate model after how many epochs of training")
    parser.add_argument("--print-freq", default=10, type=int, help="Print after iterations")
    parser.add_argument("--vis-samples", default=5, type=int, help="Number of visualization samples per epoch")


class Trainer(object):
    def __init__(self, dataset, val_dataset, args):
        self.args = args
        self.session_name = args.session_name
        self.num_epochs = self.args.epochs
        self.epoch = 0
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = self.get_network(input_channels=3 if self.args.channels == "rgb" else 12)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        self.train_losses = []
        self.val_losses = []
        self.running_loss = 0.0
        self.best_loss = np.inf

        self.model_path = f"output/{self.session_name}/models"
        self.vis_path = f"output/{self.session_name}/visualizations"
        self._create_dirs()

        print(
            f"Session initalized. Session name: {self.session_name}. Device: {self.device}.")

    def get_network(self, input_channels=3, num_classes=3):
        model = UNet(n_channels=input_channels, n_classes=num_classes)
        if self.args.model and os.path.exists(self.args.model):
            data = torch.load(self.args.model)
            model.load_state_dict(data)
            print(f"**************Checkpoint {self.args.model} loaded**************")
        return model

    def _print_epoch_stats(self, phase="train"):
        print(f"Epochs completed: {self.epoch + 1} | {phase} loss: {self.running_loss}")

    def _save_snapshot(self):
        model_path = f"{self.model_path}/best.pth"
        print(f"Best model found at epoch {self.epoch + 1}. Saving model at {model_path}")
        torch.save(self.model.state_dict(), model_path)

    def _create_dirs(self):
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.vis_path, exist_ok=True)

    def _plot_losses(self):
        x = list(range(1, self.num_epochs + 1))
        plt.plot(x, self.train_losses, "g", label="Training loss")
        plt.plot(x, self.val_losses, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{self.session_name}_losses.jpg")

    def evaluate(self):
        self.model.eval()
        dataloader = td.DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=not eval)
        print("**************Evaluating model**************")
        ious = []
        pix_accs = []
        for data in tqdm(dataloader):
            image, label = data["image"], data["label"]
            image = image.to(self.device)
            label = label.to(self.device).squeeze(dim=1).long()

            pred = self.model(image)
            pred = torch.argmax(pred.squeeze(dim=1), dim=1)

            ious.append(intersection_over_union(label.cpu().numpy(), pred.cpu().numpy()))
            pix_accs.append(pixel_accuracy(label.cpu().numpy(), pred.cpu().numpy()))

        print("**************Evaluation metrics**************")
        print(f"Average IOU: {np.mean(ious):3f}")
        print(f"Average pixel accuracy: {np.mean(pix_accs) * 100 :3f} %")

    def _save_visualizations(self, data):
        self.model.eval()
        image, label = data["image"], data["label"]
        image = image.to(self.device)
        label = label.to(self.device).squeeze(dim=1).long()
        pred = torch.argmax(self.model(image).squeeze(dim=1), dim=1)

        fig, ax = plt.subplots(figsize=(12, 6), nrows=2, ncols=self.args.vis_samples, sharey=True)

        for i in range(self.args.vis_samples):
            ax[0, i].imshow(label[i].cpu())
            ax[1, i].imshow(pred[i].cpu())
            ax[0, i].axis("off")
            ax[1, i].axis("off")

        plt.figtext(0.5, 0.95, "Ground truth")
        plt.figtext(0.5, 0.5, "Prediction")
        plt.axis("off")
        fig.tight_layout()
        plt.savefig(f"{self.vis_path}/epoch_{self.epoch + 1}.png", bbox_inches="tight")
        plt.clf()

    def train(self):
        print("**************Starting training**************")
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.running_loss = 0.0
            self.run_epoch(self.dataset)
            self.running_loss /= len(self.dataset)
            self.train_losses.append(self.running_loss)
            self._print_epoch_stats(phase="train")

            if (epoch + 1) % self.args.eval_freq == 0:
                self.validate_epoch(self.val_dataset)
                self.running_loss /= len(self.val_dataset)
                self._print_epoch_stats(phase="validation")
                self.val_losses.append(self.running_loss)
                self.evaluate_epoch(self.val_dataset)

            if self.running_loss <= self.best_loss:
                self.best_loss = self.running_loss
                self._save_snapshot()
        self._plot_losses()
        print("**************Training complete**************")

    def validate_epoch(self, dataset):
        print("**************Starting validation**************")
        self.model.eval()
        self.running_loss = 0.0
        self.run_epoch(dataset, eval=True)

    def run_epoch(self, dataset, eval=False):
        dataloader = td.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=not eval)
        for i, data in enumerate(dataloader):
            self.run_batch(data, eval, i)

        if eval:
            self._save_visualizations(data)

    def run_batch(self, data, eval, idx):
        self.model.train(not eval)
        with torch.set_grad_enabled(not eval):
            image, label = data["image"], data["label"]
            image = image.to(self.device)
            label = label.to(self.device).squeeze(dim=1).long()

            pred = self.model(image)
            pred = pred.squeeze(dim=1)

            loss = self.criterion(F.log_softmax(pred, 1), label)
            self.running_loss += loss.detach().cpu().item()

            if (idx % self.args.print_freq) == 0:
                print(f"Epoch: {self.epoch + 1} | Iter: {idx} | Loss: {loss.item():5f}")
            if not eval:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
