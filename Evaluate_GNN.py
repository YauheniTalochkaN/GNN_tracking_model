#!/usr/bin/env python

import argparse
import yaml
import os
from itertools import product
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from GNN_training import EdgeClassificationGNN, load_npz_to_pyg
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/training_parameters.yaml')
    return parser.parse_args()

# Evaluate metrics
def evaluate(model, loader, device, thresholds):
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    purities = []
    efficiencies = []
    accuracies = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            edge_logits = model(data)
            all_true_labels.append(data.y.cpu().numpy())
            all_pred_labels.append(edge_logits.cpu().numpy())

    all_true_labels = np.concatenate(all_true_labels)
    all_pred_labels = np.concatenate(all_pred_labels)
    
    for threshold in thresholds:
        binary_preds = (all_pred_labels >= threshold).astype(int)
        true_positive = np.sum((binary_preds == 1) & (all_true_labels == 1))
        true_negative = np.sum((binary_preds == 0) & (all_true_labels == 0))
        false_positive = np.sum((binary_preds == 1) & (all_true_labels == 0))
        false_negative = np.sum((binary_preds == 0) & (all_true_labels == 1))

        purity = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        efficiency = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

        purities.append(purity)
        efficiencies.append(efficiency)
        accuracies.append(accuracy)
    
    return all_true_labels, all_pred_labels, purities, efficiencies, accuracies

def plot_purity_and_efficiency(purities, efficiencies, accuracies, thresholds):
    # Plot Purity, Efficiency and Accuracy
    purities = [x * 100 for x in purities]
    efficiencies = [x * 100 for x in efficiencies]
    accuracies = [x * 100 for x in accuracies]

    plt.close('all')
    plt.figure(figsize=(7, 5))
    if thresholds[-1] >= 1:
        plt.plot(thresholds[:-1], purities[:-1], alpha=1, label='Purity', color='blue')
    else: 
        plt.plot(thresholds, purities, alpha=1, label='Purity', color='blue')
    plt.plot(thresholds, efficiencies, alpha=1, label='Efficiency', color='orange')
    plt.plot(thresholds, accuracies, alpha=1, label='Accuracy', color='red')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-2, 102])
    plt.xlabel('Cut on model score', fontsize=16)
    plt.ylabel('Metrics (%)', fontsize=16)
    plt.tick_params(axis='both', 
                    which='major', 
                    labelsize=14,
                    top=True,
                    bottom=True,
                    left=True,
                    right=True,
                    direction='in')
    plt.legend(loc='center', fontsize=16)
    plt.show()
    #plt.savefig("Metrics.png")

def plot_ROC(fpr, tpr, roc_auc):
    # Plot ROC Curve
    plt.close('all')
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='orange', alpha=1)
    plt.plot([0, 1], [0, 1], color='blue', alpha=1, linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.tick_params(axis='both', 
                    which='major', 
                    labelsize=14,
                    top=True,
                    bottom=True,
                    left=True,
                    right=True,
                    direction='in')
    plt.title(f'ROC curve (AUC = {roc_auc:.3f})', fontsize=16)
    plt.show()
    #plt.savefig("ROC.png")

def plot_counts(fake_pred, true_pred):
    # Plot the first histogram with log scale
    plt.close('all')
    plt.figure(figsize=(7, 5))
    plt.hist(fake_pred, bins=100, alpha=1, label='fake', log=True, histtype='step', color='orange')
    plt.hist(true_pred, bins=100, alpha=1, label='true', log=True, histtype='step', color='blue')
    plt.xlim([-0.02, 1.02])
    #plt.ylim([1e2, 1e7])
    plt.xlabel('Model output', fontsize=16)
    plt.ylabel('Counts', fontsize=16)
    plt.tick_params(axis='both', 
                    which='major', 
                    labelsize=14,
                    top=True,
                    bottom=True,
                    left=True,
                    right=True,
                    direction='in')
    plt.tick_params(axis='y',          
                    which='minor', 
                    left=False )
    plt.legend(fontsize=16, loc='upper center')
    plt.show()
    #plt.savefig("Counts.png")

def main():
    # Get args
    args = parse_args()

    # Open the file of training parameters
    with open(args.config, 'r') as f:
        parameters = yaml.safe_load(f)

    # Obtain necessary parameters
    input_dir = parameters['input_dir']
    output_dir = parameters['output_dir']
    n_files = parameters['n_files']
    section_num = parameters['section_num']
    test_size = parameters['test_size']
    batch_size = parameters['batch_size']
    n_iters = parameters['n_iters']

    # Load data
    graph_files = [os.path.join(input_dir, f'event_{evtid}_section_{section_id}_graph.npz') for evtid, section_id in product(range(n_files), range(section_num))]
    dataset = [load_npz_to_pyg(file) for file in graph_files]

    # Split data into train and test sets
    train_len = int(len(dataset) * (1-test_size))
    test_dataset = dataset[train_len:]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    thresholds = np.linspace(0, 1, 1000)
    all_true_labels, all_pred_labels, purities, efficiencies, accuracies = evaluate(model, test_loader, device, thresholds)
    plot_purity_and_efficiency(purities, efficiencies, accuracies, thresholds)

    # Calculate ROC Curve and AUC
    fpr, tpr, _ = roc_curve(all_true_labels, all_pred_labels)
    roc_auc = auc(fpr, tpr)
    plot_ROC(fpr, tpr, roc_auc)

    # Separate predictions into true and fake categories
    true_pred = all_pred_labels[all_true_labels == 1]
    fake_pred = all_pred_labels[all_true_labels == 0]
    plot_counts(fake_pred, true_pred)

if __name__ == '__main__':
    main()