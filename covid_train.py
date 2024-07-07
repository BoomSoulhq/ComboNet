import os, random, math
import copy
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from rdkit.Chem import Descriptors
from rdkit import Chem
from tqdm import trange

from chemprop.parsing import add_train_args, modify_train_args
from chemprop.models import MoleculeModel
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, split_data
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, \
    makedirs, save_checkpoint
from chemprop.train import evaluate, evaluate_predictions, predict

torch.backends.cudnn.benchmark = False


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps`
        steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(
            max(1.0, self.t_total - self.warmup_steps)))


class DiseaseModel(nn.Module):

    def __init__(self, args):
        super(DiseaseModel, self).__init__()
        self.encoder = MoleculeModel(classification=False, multiclass=False)
        self.encoder.create_encoder(args)
        self.encoder.ffn = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.latent_size)
        )
        self.covid_ffn = nn.Linear(args.latent_size, args.num_tasks)
        self.ffn = self.covid_ffn

        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def DTI_forward(self, smiles_batch):
        DTI_vecs = self.encoder(smiles_batch)
        DTI_vecs = torch.sigmoid(DTI_vecs)  # covid is zero
        return DTI_vecs

    def forward(self, smiles_batch):
        DTI_vecs = self.DTI_forward(smiles_batch)

        return self.ffn(DTI_vecs)



def prepare_data(args):
    dti_data = get_data(path=args.dti_path, args=args)
    args.use_compound_names = True
    covid_data = get_data(path=args.covid_path, args=args)
    args.use_compound_names = False

    args.output_size = len(dti_data[0].targets)
    args.num_tasks = 1
    args.train_data_size = len(covid_data)

    return dti_data, covid_data


def train(dti_data, covid_data, model, optimizer, scheduler, loss_func, args):
    model.train()
    covid_data.shuffle()
    dti_data.shuffle()

    for i in trange(0, len(covid_data), args.batch_size):
        model.zero_grad()
        covid_batch = MoleculeDataset(covid_data[i:i + args.batch_size])
        dti_batch = MoleculeDataset(dti_data[i:i + args.batch_size])
        if len(covid_batch) < args.batch_size:
            continue

        # DTI batch
        smiles, targets = dti_batch.smiles(), dti_batch.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in targets])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in targets])
        preds = model.encoder(smiles)[:, :targets.size(1)]
        dti_loss = loss_func(preds, targets)
        dti_loss = (dti_loss * mask).sum() / mask.sum()
        smiles = targets = mask = None

        smiles, targets = covid_batch.smiles(), covid_batch.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in targets])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in targets])
        preds = model(smiles)
        covid_loss = loss_func(preds, targets)
        covid_loss = (covid_loss * mask).sum() / mask.sum()
        smiles = targets = mask = None

        loss = args.dti_lambda * dti_loss + args.single_lambda * covid_loss
        loss.backward()
        optimizer.step()
        scheduler.step()


def combo_evaluate(model, data, args):
    model.eval()
    all_preds, all_targets = [], []

    for i in trange(0, len(data), args.batch_size):
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles = mol_batch.smiles()
        targets = mol_batch.targets()
        preds = model(smiles)
        all_preds.extend(preds.tolist())
        all_targets.extend(targets)

    score = evaluate_predictions(all_preds, all_targets, args.num_tasks, args.metric_func, args.dataset_type)
    return score


def run_training(args, save_dir):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dti_data, covid_data = prepare_data(args)

    model = DiseaseModel(args)
    loss_func = get_loss_func(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=args.train_data_size / args.batch_size * 2,
                                     t_total=args.train_data_size / args.batch_size * args.epochs
                                     )

    args.metric_func = get_metric_func(metric=args.metric)
    best_score = float('inf') if args.minimize_score else -float('inf')
    best_epoch = 0

    for epoch in range(1):
        print(f'Epoch {epoch}')
        train(dti_data, covid_data, model, optimizer, scheduler, loss_func, args)

        val_scores = combo_evaluate(model, covid_data, args)
        avg_val_score = np.nanmean(val_scores)
        print(f'Combo Validation {args.metric} = {avg_val_score:.4f}')

        # only save checkpoints when DTI prediction is accurate enough (after five epochs)
        if epoch >= 0 and (
                args.minimize_score and avg_val_score < best_score or not args.minimize_score and avg_val_score > best_score):
            best_score, best_epoch = avg_val_score, epoch
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, args=args)

    print(f'Loading model checkpoint from epoch {best_epoch}')
    ckpt_path = os.path.join(save_dir, 'model.pt')
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])

    test_scores = combo_evaluate(model, covid_data, args)
    avg_test_scores = np.nanmean(test_scores)
    print(f'Test {args.metric} = {avg_test_scores:.4f}')

    return avg_test_scores


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dti_path', default="data/covid/dti.csv")
    parser.add_argument('--covid_path', default="data/covid/single_agent.csv")
    parser.add_argument('--single_lambda', type=float, default=0.1)
    parser.add_argument('--combo_lambda', type=float, default=1)
    parser.add_argument('--dti_lambda', type=float, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_size', type=int, default=100)
    parser.add_argument('--num_covid_targets', type=int, default=35)

    add_train_args(parser)
    args = parser.parse_args()
    args.data_path = 'data/hiv/hiv.csv'
    args.dataset_type = 'classification'
    args.save_dir = 'data/trained_model'
    args.num_folds = 3

    modify_train_args(args)
    print(args)

    all_test_scores = [0] * args.num_folds
    for i in range(0, args.num_folds):
        fold_dir = os.path.join(args.save_dir, f'fold_{i}')
        makedirs(fold_dir)
        args.seed = i
        all_test_scores[i] = run_training(args, fold_dir)

    all_test_scores = np.stack(all_test_scores, axis=0)
    mean, std = np.mean(all_test_scores, axis=0), np.std(all_test_scores, axis=0)
    print(f'{args.num_folds} fold average: {mean} +/- {std}')
