import os, random, sys
import time

import numpy as np
from argparse import ArgumentParser

import pandas as pd
import torch

from chemprop.data.utils import get_data
from covid_train import DiseaseModel


def combo_evaluate(model, dti_data, args):
    model.eval()
    all_preds = []
    for i in range(5):  # len(dti_data)
        preds = model.pre_forward([dti_data.data[i].smiles], mode=0)
        all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--test_path', default=r"D:\users\p30057372\ComboNet\ComboNet-master\data\covid\dti.csv")
    parser.add_argument('--checkpoint_dir', default=r"D:\users\p30057372\ComboNet\ComboNet-master\data\fold_0")
    args = parser.parse_args()

    data = get_data(path=args.test_path)

    args.checkpoint_paths = []
    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname.endswith('.pt'):
                args.checkpoint_paths.append(os.path.join(root, fname))

    sum_preds = np.zeros((len(data),))
    with torch.no_grad():
        pred_output = pd.DataFrame()
        for checkpoint_path in args.checkpoint_paths:
            ckpt = torch.load(checkpoint_path)
            ckpt['args'].attention = False
            model = DiseaseModel(ckpt['args'])
            model.load_state_dict(ckpt['state_dict'])
            model_preds = combo_evaluate(model, data, ckpt['args'])
            df = pd.DataFrame(model_preds)
            pred_output = pd.concat([pred_output, df], ignore_index=True)
        pred_output.to_csv(r"D:\users\p30057372\ComboNet\ComboNet-master\data\测试结果{now}.csv".format(
            now=time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))))
