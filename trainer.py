import os
import gc
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchmetrics

from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
            self,
            log_dir,
            model,
            train_ds,
            test_ds,
            batch_size,
            epochs,
            ext_eval_datasets: list = None,
            eval_every=1,
            device='cpu',
            optimizer=None,
            save_best=False,
            eval_after_epoch=0,
            binary_classification_regime=False,
    ):  
        self.binary_classification_regime = binary_classification_regime
        self.ext_eval_datasets = ext_eval_datasets
        self.device = device
        self.eval_every = eval_every
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_ds = test_ds
        self.train_ds = train_ds
        self.model = model
        self.log_dir = log_dir
        self.best_acc = 0.
        self.save_best = save_best
        self.eval_after_epoch = eval_after_epoch

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters())
        else:
            self.optimizer = optimizer
        self.global_epoch = 0
        self.tensorboard_writer = SummaryWriter(os.path.join(self.log_dir, 'tensorboard'))

    def train_epoch(self, epoch):
        self.model.train()
        with tqdm(total=len(self.train_ds), desc='epoch {} of {}'.format(epoch, self.epochs)) as pbar:
            epoch_ce_loss, epoch_acc, epoch_recon_loss = 0, 0, 0

            for i, batch in enumerate(self.train_ds):
                img, label = batch
                if isinstance(label, list):
                    label = label[0]
                img, label = img.to(self.device), label.to(self.device)
                out = self.model(img)

                loss = F.cross_entropy(out, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(
                    ce_loss='{:.3f}'.format(loss.item()),
                )
                pbar.update()

                epoch_ce_loss += loss

            self.tensorboard_writer.add_scalar("Training/ce_loss", epoch_ce_loss / len(self.train_ds), self.global_epoch)
            print(f"Training data after epoch {self.global_epoch}: ce loss - {epoch_ce_loss / len(self.train_ds)} ")

    @torch.no_grad()
    def eval(self, dataset=None, dataset_name='test'):
        self.model.eval()
        if dataset is None:
            dataset = self.test_ds

        epoch_ce_loss, correct = 0, 0
        total = 0
        precision, recall = 0., 0.

        for i, batch in enumerate(dataset):
            img, label = batch
            if isinstance(label, list):
                label = label[0]
            img, label = img.to(self.device), label.to(self.device)
            out = self.model(img, labels=label)

            ce_loss = F.cross_entropy(out, label)

            epoch_ce_loss += ce_loss
            correct += torch.sum(out.argmax(-1) == label)
            total += img.shape[0]

            if len(out.shape) < 2:
                precision += 100 * torchmetrics.functional.precision(out.argmax(-1), label, 'binary')
                recall += 100 * torchmetrics.functional.recall(out.argmax(-1), label, 'binary')

        total_acc = (correct / total) * 100
        precision /= i+1
        recall /= i+1
        self.tensorboard_writer.add_scalar(f"{dataset_name}/ce_loss", epoch_ce_loss / i, self.global_epoch)
        self.tensorboard_writer.add_scalar(f"{dataset_name}/acc", total_acc, self.global_epoch)
        if len(out.shape) < 2:
            self.tensorboard_writer.add_scalar(f"{dataset_name}/recall", recall, self.global_epoch)
            self.tensorboard_writer.add_scalar(f"{dataset_name}/precision", precision, self.global_epoch)

        print(f"\tDataset: {dataset_name}")
        print(f"\t--- ce loss - {epoch_ce_loss / (i+1)} ")
        print(f"\t--- accuracy - {total_acc} ")
        print(f"\t--- precision - {precision} ")
        print(f"\t--- recall - {recall} ")
        print("\t======================================")
        return total_acc

    def train(self, eval_only=False):
        for epoch in range(1, self.epochs+1):
            self.global_epoch += 1

            if not eval_only:
                self.train_epoch(self.global_epoch)

            if epoch % self.eval_every == 0 and epoch > self.eval_after_epoch:
                self.eval(self.test_ds)

                if self.ext_eval_datasets is not None:
                    avg_acc = 0.
                    for (ds_name, ds) in self.ext_eval_datasets:
                        acc = self.eval(ds, dataset_name=ds_name)
                        avg_acc += acc
                    avg_acc /= len(self.ext_eval_datasets)
                    if self.save_best and avg_acc > self.best_acc:
                        self.best_acc = avg_acc
                        self._save_checkpoint(os.path.join(self.log_dir, "best_ckpt.pt"))

    def _save_checkpoint(self, path=None):
        checkpoint = {'model': self.model.state_dict(), 'global_epoch': self.global_epoch}

        if path is not None:
            torch.save(checkpoint, os.path.join(path))
        else:
            torch.save(checkpoint, os.path.join(self.log_dir, 'model_ckpt.pt'))

    def restore_model_from_checkpoint(self, path=None, device=None):
        if path is not None:
            checkpoint = torch.load(path, map_location=device)
        else:
            checkpoint = torch.load(os.path.join(self.log_dir, "ckpt.pt"), map_location=device)

        self.model.load_state_dict(checkpoint['model'])
        if device is not None:
            self.model.to(device)

        self.global_epoch = checkpoint['global_epoch']

        print(f"Continuing the experiment from epoch {self.global_epoch+1}...")


class ArtDatasetCustomTrainer(Trainer):
    def train_epoch(self, epoch):
        self.model.train()
        with tqdm(total=len(self.train_ds), desc='epoch {} of {}'.format(epoch, self.epochs)) as pbar:
            epoch_ce_loss, correct, total = 0, 0, 0

            for i, batch in enumerate(self.train_ds):
                img, label = batch
      
                img, label = img.to(self.device), label.to(self.device)
                if len(img.shape) > 4:
                    out = self.model(img.flatten(0, 1))
                    out = out.unflatten(0, (-1, img.shape[1]))
                else: 
                    out = self.model(img)

                if len(out.shape) > 1:
                    if self.binary_classification_regime:
                        num_classes = out.shape[1]
                        loss = F.binary_cross_entropy_with_logits(
                            out.flatten(), F.one_hot(label, num_classes=num_classes).float().flatten(),
                            pos_weight=(num_classes - 1) * torch.ones_like(out.flatten()))
                    else:
                        loss = F.cross_entropy(out, label)
                else:
                    loss = F.binary_cross_entropy_with_logits(out, label.float())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(
                    ce_loss='{:.3f}'.format(loss.item()),
                )
                pbar.update()

                epoch_ce_loss += loss

                if len(out.shape)>1:
                    correct += torch.sum(out.argmax(-1) == label)
                else:
                    correct += torch.sum((out.sigmoid() > 0.5).long() == label)
                total += img.shape[0]
            
            total_acc = (correct / total) * 100

            self.tensorboard_writer.add_scalar("Training/ce_loss", epoch_ce_loss / len(self.train_ds), self.global_epoch)
            self.tensorboard_writer.add_scalar(f"Training/acc", total_acc, self.global_epoch)
            print(f"Training data after epoch {self.global_epoch}: ce loss - {epoch_ce_loss / len(self.train_ds)} ")

    @torch.no_grad()
    def eval(self, dataset=None, dataset_name='test', reinit_writer=False):
        self.model.eval()
        if dataset is None:
            dataset = self.test_ds

        epoch_ce_loss, correct = 0, 0
        total = 0
        precision, recall = 0., 0.

        for i, batch in enumerate(dataset):
            img, label = batch
            img, label = img.to(self.device), label.to(self.device)
            if len(img.shape) > 4:
                out = self.model(img.flatten(0, 1))
                out = out.unflatten(0, (-1, img.shape[1]))
            else: 
                out = self.model(img)

            ce_loss = F.cross_entropy(out, label.long() if len(out.shape)>1 else label.float())
            
            epoch_ce_loss += ce_loss
            if len(out.shape)>1:
                correct += torch.sum(out.argmax(-1) == label)
            else:
                correct += torch.sum((out.sigmoid() > 0.5).long() == label)
            total += img.shape[0]

        total_acc = (correct / total) * 100
        precision /= i+1
        recall /= i+1
        if reinit_writer:
            self.tensorboard_writer = SummaryWriter(os.path.join(self.log_dir, 'tensorboard'))
        self.tensorboard_writer.add_scalar(f"{dataset_name}/ce_loss", epoch_ce_loss / i, self.global_epoch)
        self.tensorboard_writer.add_scalar(f"{dataset_name}/acc", total_acc, self.global_epoch)

        print(f"\tDataset: {dataset_name}")
        print(f"\t--- ce loss - {epoch_ce_loss / (i+1)} ")
        print(f"\t--- accuracy - {total_acc} ")
        print("\t======================================")

        if self.save_best and total_acc > self.best_acc:
            self.best_acc = total_acc
            self._save_checkpoint(os.path.join(self.log_dir, "best_ckpt.pt"))

        return total_acc
