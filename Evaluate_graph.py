#!/usr/bin/env python

import argparse
import os
import time
import yaml
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from GNN_training import EdgeClassificationGNN
from plot_graph import get_pos, get_3Dpos, load_graph_from_npz
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def convert_nx_to_pyg_data(G):
    # Get features of nodes and edges
    x = np.array([G.nodes[node]['pos'] for node in G.nodes])
    edge_index = np.array(list(G.edges)).T
    edge_attr = np.array([G.edges[edge]['features'] for edge in G.edges])
    edge_labels = np.array([G.edges[edge]['label'] for edge in G.edges])
    
    # Convert them into torch tensors
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_labels = torch.tensor(edge_labels, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)

# Evaluate metrics
def evaluate(model, loader, device, threshold=0.5):
    start_time = time.time()
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            edge_logits = model(data)
            all_true_labels.append(data.y.cpu().numpy())
            all_pred_labels.append((edge_logits >= threshold).cpu().numpy())

    all_true_labels = np.concatenate(all_true_labels)
    all_pred_labels = np.concatenate(all_pred_labels)
    
    true_positive = np.sum((all_pred_labels == 1) & (all_true_labels == 1))
    true_negative = np.sum((all_pred_labels == 0) & (all_true_labels == 0))
    false_positive = np.sum((all_pred_labels == 1) & (all_true_labels == 0))
    false_negative = np.sum((all_pred_labels == 0) & (all_true_labels == 1))
    
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    purity = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    efficiency = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    print(f'Accuracy: {accuracy:.4f}, Purity: {purity:.4f}, Efficiency: {efficiency:.4f}')

    print(f"Spent time: {time.time() - start_time:.6f} s")
    
    return all_pred_labels

def plot_graph(G, pred):
    n_edges = len(G.edges())
    edge_colors = ['green']*n_edges
    edge_alpha = [1.]*n_edges
    for iedge,edge in enumerate(G.edges(data=True)):
        if int(edge[2]['label']) != 1:
            if int(edge[2]['label']) == int(pred[iedge]):
                edge_colors[iedge] = 'grey'
                edge_alpha[iedge] = 0.3
            else:
                edge_colors[iedge] = 'indigo'
                edge_alpha[iedge] = 0.5
        else:
            if int(edge[2]['label']) != int(pred[iedge]):
                edge_colors[iedge] = 'red'
    pos = get_pos(G) 
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    widths = [2.0 if color == 'red' or 'green' else 1.5 for color in edge_colors]
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=3, alpha=1.0, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=widths, alpha=edge_alpha, ax=ax, arrows=False) 
    plt.show()

def plot3D_networkx(G, pred, animate=False, only_true=False):
    
    pos = get_3Dpos(G)
   
    x = [pos[i][0] for i in pos]
    y = [pos[i][1] for i in pos]
    z = [pos[i][2] for i in pos]

    plt.close('all')

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, s=3, c='black')

    for iedge,edge in enumerate(G.edges(data=True)):
        p1 = G.nodes[edge[0]]['pos'][:3]
        p2 = G.nodes[edge[1]]['pos'][:3]
        col = 'green'
        alpha = 1
        linewidth = 1
        if int(edge[2]['label']) != 1:
            if int(edge[2]['label']) == int(pred[iedge]):
                col = 'grey'
                alpha = 0.5
                linewidth = 0.7
            else:
                col = 'indigo'
                alpha = 1
                linewidth = 0.7
        else:
            if int(edge[2]['label']) != int(pred[iedge]):
                col = 'red'
        if int(edge[2]['label']) == 1:
            ax.plot([p1[0] * np.cos(p1[1]), p2[0] * np.cos(p2[1])], 
                    [p1[0] * np.sin(p1[1]), p2[0] * np.sin(p2[1])],
                    [p1[2], p2[2]], color=col, linewidth=linewidth, alpha=alpha)
        elif not only_true:
            ax.plot([p1[0] * np.cos(p1[1]), p2[0] * np.cos(p2[1])], 
                    [p1[0] * np.sin(p1[1]), p2[0] * np.sin(p2[1])],
                    [p1[2], p2[2]], color=col, linewidth=linewidth, alpha=alpha)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.set_zlabel('Z', fontsize=16)
    ax.set_title('', fontsize=16)

    plt.show()

    if animate:
        def animate(angle):
            ax.view_init(elev=30., azim=angle)
            return fig,

        angle = np.linspace(0, 360, 240, endpoint=False)
        ani = FuncAnimation(fig, animate, angle, interval=150, blit=True)

        ani.save('3D_graph_animation.gif', writer=PillowWriter(fps=30))

def parse_args():
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('graph_file')
    add_arg('config', nargs='?', default='configs/training_parameters_mpd.yaml')
    add_arg('--threshold', type=float, default=0.5)
    add_arg('--animate', type=str, default='False')
    add_arg('--only_true', type=str, default='False')
    return parser.parse_args()

def str_to_bool(s):
    if s == 'True' or s == 'true':
        return True
    else: 
        return False

def main():
    # Get args
    args = parse_args()

    # Load graph
    G = load_graph_from_npz(args.graph_file)
    
    # Get threshold
    threshold = args.threshold

    # Open the file of training parameters
    with open(args.config, 'r') as f:
        parameters = yaml.safe_load(f)

    # Obtain necessary parameters
    output_dir = parameters['output_dir']
    n_iters = parameters['n_iters']

    # Convert graph to Data
    dataset = [convert_nx_to_pyg_data(G)]

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
            print(f'Loaded checkpoint from epoch {checkpoint['epoch']+1}')
        else:
            print('Checkpoint file is missing some keys, starting from scratch.')
            return
    
    # Build loader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Evaluate graph and get predictions
    pred = evaluate(model, loader, device, threshold)
    
    # Plot graph
    plot_graph(G, pred)

    # Plot 3D
    plot3D_networkx(G, pred, str_to_bool(args.animate), str_to_bool(args.only_true))

if __name__ == '__main__':
    main()