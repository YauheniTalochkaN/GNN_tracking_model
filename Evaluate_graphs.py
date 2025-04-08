#!/usr/bin/env python

import argparse
import yaml
import os
import time
from itertools import product
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from GNN_training import EdgeClassificationGNN


def parse_args():
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/training_parameters.yaml')
    add_arg('--threshold', type=float, default=0.5)
    return parser.parse_args()

def load_npz_to_pyg(filename):
    with np.load(filename, allow_pickle=True) as data:
        x = torch.from_numpy(data['node_pos']).float()
        hit_id = torch.from_numpy(data['node_hit_id']).long()
        edge_index = torch.from_numpy(data['edges']).long().t().contiguous()
        edge_attr = torch.from_numpy(data['edge_features']).float()
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, hit_id=hit_id)

# Evaluate metrics
def evaluate(model, loader, device, threshold = 0.5):
    start_time = time.time()
    model.eval()

    if not os.path.exists('./answers'):
        os.makedirs('./answers')
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc="Evaluating", unit="batch")):
            data = data.to(device)
            edge_logits = model(data)

            true_edges = data.edge_index[:, edge_logits >= threshold]

            in_hits = data.hit_id[true_edges[0]]
            out_hits = data.hit_id[true_edges[1]]

            result = torch.stack([in_hits, out_hits], dim=1).cpu().numpy()

            df = pd.DataFrame(result, columns=['in_hit_id', 'out_hit_id'])

            df.to_csv(f'answers/answer_{i}.csv', index=False)
    
    print(f"Spent time: {time.time() - start_time:.6f} s")

def main():
    # Get args
    args = parse_args()

    # Get threshold
    threshold = args.threshold

    # Open the file of training parameters
    with open(args.config, 'r') as f:
        parameters = yaml.safe_load(f)

    # Obtain necessary parameters
    input_dir = parameters['input_dir']
    output_dir = parameters['output_dir']
    n_files = parameters['n_files']
    section_num = parameters['section_num']
    test_size = parameters['test_size']
    n_iters = parameters['n_iters']

    # Load data
    graph_files = [os.path.join(input_dir, f'event_{evtid}_section_{section_id}_graph.npz') for evtid, section_id in product(range(n_files), range(section_num))]
    dataset = [load_npz_to_pyg(file) for file in graph_files]

    # Split data into train and test sets
    train_len = int(len(dataset) * (1-test_size))
    test_dataset = dataset[train_len:]

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device, "device")

    # Initializing the model
    node_feature_dim = dataset[0].x.size(1)
    edge_feature_dim = dataset[0].edge_attr.size(1)
    model = EdgeClassificationGNN(node_feature_dim, edge_feature_dim, n_iters).to(device) 

    # The folder where the model was saved
    model_save_path = os.path.join(output_dir, 'model_checkpoint.pth')

    # Load the existing checkpoint
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, weights_only=True, map_location=device)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'epoch' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Loaded checkpoint from epoch {checkpoint["epoch"]+1}')
        else:
            print('Checkpoint file is missing some keys, starting from scratch.')
            return

    # Evaluate model
    evaluate(model, test_loader, device, threshold)

if __name__ == '__main__':
    main()