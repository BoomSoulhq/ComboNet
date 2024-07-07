import os,
import time

import numpy as np
from argparse import ArgumentParser

import pandas as pd
import torch

from covid_train import DiseaseModel


def combo_evaluate(model, data, args):
    model.eval()
    all_preds = []
    smile_representations = []
    for i in range(0, len(data), args.batch_size):
        mol_batch = data[i: i + args.batch_size]
        smiles = [smile[0] for smile in mol_batch]
        cur_smile_representations = model.DTI_forward(smiles).cpu().numpy()
        preds = torch.sigmoid(model(smiles)).round().cpu().numpy()
        ans = np.concatenate((cur_smile_representations, preds), axis=1)

        all_preds.append(ans)
    return np.concatenate(all_preds, axis=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--test_path', default='data/covid/synergy_experiment.csv')
    parser.add_argument('--checkpoint_dir', default="data/trained_model")
    args = parser.parse_args()
    args.test_path = 'data/covid/synergy_experiment.csv'

    with open(args.test_path) as f:
        header = next(f)
        data = [line.strip("\r\n ").split(',')[:1] for line in f]

    args.checkpoint_paths = []
    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname.endswith('.pt'):
                args.checkpoint_paths.append(os.path.join(root, fname))

    sum_preds = np.zeros((len(data),))
    with torch.no_grad():
        for checkpoint_path in args.checkpoint_paths:
            ckpt = torch.load(checkpoint_path)
            ckpt['args'].attention = False
            model = DiseaseModel(ckpt['args'])
            feature_weights = model.ffn.weight.numpy()
            feature_weights = np.append(feature_weights[0], ["label"]).reshape(1, 101)
            model.load_state_dict(ckpt['state_dict'])
            model_preds = combo_evaluate(model, data, ckpt['args'])
            pred_output = pd.DataFrame(np.concatenate((feature_weights, model_preds), axis=0))
            pred_output.index = list(["weight"] + [x[0] for x in data])
            pred_output.to_csv(r"D:\users\p30057372\ComboNet\ComboNet-master\data\测试结果{now}.csv".format(
                now=time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))))
