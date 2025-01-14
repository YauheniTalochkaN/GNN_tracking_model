#!/usr/bin/env python

import argparse
import yaml
import os
import time
import logging
from itertools import product
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


def parse_args():
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/training_parameters_mpd.yaml')
    return parser.parse_args()

def load_npz_to_pyg(filename):
    with np.load(filename, allow_pickle=True) as data:
        x = torch.from_numpy(data['node_pos']).float()
        edge_index = torch.from_numpy(data['edges']).long().t().contiguous()
        edge_attr = torch.from_numpy(data['edge_features']).float()
        y = torch.from_numpy(data['edge_labels']).float()
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

class CustomMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=torch.nn.Tanh(), end_activation=None, dropout=torch.nn.Dropout(0.1), use_layer_norm=True):
        super(CustomMLP, self).__init__()
        layers = []
        # Добавляем первый скрытый слой
        layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        if use_layer_norm:
            layers.append(torch.nn.LayerNorm(hidden_sizes[0])) 
        layers.append(activation)
        layers.append(dropout)
        
        # Добавляем остальные скрытые слои
        for i in range(1, len(hidden_sizes)):
            layers.append(torch.nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            if use_layer_norm:
                layers.append(torch.nn.LayerNorm(hidden_sizes[i]))
            layers.append(activation)
            layers.append(dropout)
        
        # Добавляем выходной слой
        layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))
        if end_activation is not None:
            layers.append(end_activation)

        # Объединяем все слои в nn.Sequential
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
""" class CustomWeightedGATConv(MessagePassing):
    def __init__(self, node_hidden_dim, hidden_dims, output_dim, edge_hidden_dim, node_feature_dim, activation=torch.nn.Tanh(), end_activation=None, dropout=torch.nn.Dropout(0.1)):
        super(CustomWeightedGATConv, self).__init__(aggr='add')
        self.mlp = CustomMLP(2 * node_hidden_dim + node_feature_dim + edge_hidden_dim, hidden_dims, output_dim, activation, end_activation, dropout)

    def forward(self, x, edge_index, edge_attr, edge_weight, initial_x):
        # Прямой проход
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight, initial_x=initial_x)

    def message(self, x_j, edge_attr, edge_weight):
        # Умножаем сообщения на веса рёбер и добавляем особенности рёбер
        return edge_weight.view(-1, 1) * torch.cat([x_j, edge_attr], dim=-1)

    def update(self, aggr_out, x, initial_x):
        # Объединяем агрегированные особенности соседей с собственными особенностями узлов
        out = torch.cat([x, initial_x, aggr_out], dim=-1)
        # Пропускаем через MLP
        return self.mlp(out) """
    
class CustomWeightedGATConv(MessagePassing):
    def __init__(self, node_hidden_dim, hidden_dims1, hidden_dims2, output_dim, edge_hidden_dim, node_feature_dim, activation=torch.nn.Tanh(), end_activation=None, dropout=torch.nn.Dropout(0.1)):
        super(CustomWeightedGATConv, self).__init__(aggr='add')
        self.x_mlp = CustomMLP(3 * node_hidden_dim + node_feature_dim + 2 * edge_hidden_dim, hidden_dims1, output_dim, activation, end_activation, dropout)
        self.weight_mlp = CustomMLP(edge_hidden_dim + 1, hidden_dims2, 1, activation, end_activation, dropout)

    def forward(self, x, edge_index, edge_attr, edge_weight, initial_x):        
        # Первый шаг агрегации (1-hop)
        one_hop_out = self.propagate(edge_index, y=x, z=x, edge_attr=edge_attr, edge_weight=edge_weight, hop=1)

        # Второй шаг агрегации (2-hop), используя результаты первого шага
        two_hop_out = self.propagate(edge_index, y=one_hop_out, z=x, edge_attr=edge_attr, edge_weight=edge_weight, hop=2)

        # Передаем результаты в update для финального объединения
        return self.final_update(x=x, one_hop_out=one_hop_out, two_hop_out=two_hop_out, initial_x=initial_x)

    def message(self, y_j, z_i, edge_attr, edge_weight, hop):

        weights = self.weight_mlp(torch.cat([edge_weight.view(-1, 1), edge_attr], dim=-1))

        # Умножаем сообщения на веса рёбер и добавляем особенности рёбер
        if hop == 1:
            return weights * torch.cat([y_j, edge_attr], dim=-1)
        elif hop == 2:
            return weights * (y_j - weights * torch.cat([z_i, edge_attr], dim=-1))

    def update(self, aggr_out):
        return aggr_out
    
    def final_update(self, x, one_hop_out, two_hop_out, initial_x):
        # Объединяем агрегированные особенности соседей с собственными особенностями узлов
        out = torch.cat([x, initial_x, one_hop_out, two_hop_out], dim=-1)
        # Пропускаем через MLP
        return self.x_mlp(out)

# Model class
class EdgeClassificationGNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, n_iters=6, node_hidden_dim=64, edge_hidden_dim=64):
        super(EdgeClassificationGNN, self).__init__()

        self.n_iters = n_iters

        self.node_encoder = CustomMLP(node_feature_dim, [128, 128], node_hidden_dim)
        self.edge_encoder = CustomMLP(2 * node_feature_dim + edge_feature_dim, [128, 128], edge_hidden_dim)
        self.initial_edge_classification_mlp = CustomMLP(2 * node_hidden_dim + edge_hidden_dim + 2 * node_feature_dim + edge_feature_dim, [128, 128], 1, end_activation=torch.nn.Sigmoid())
        self.node_gatconv = CustomWeightedGATConv(node_hidden_dim, [256, 128], [128, 128], node_hidden_dim, edge_hidden_dim, node_feature_dim)
        self.edge_mlp= CustomMLP(2 * node_hidden_dim + edge_hidden_dim + 1 + 2 * node_feature_dim + edge_feature_dim, [128, 128], edge_hidden_dim)
        self.edge_classification_mlp = CustomMLP(2 * node_hidden_dim + edge_hidden_dim + 1 + 2 * node_feature_dim + edge_feature_dim, [128, 128], 1, end_activation=torch.nn.Sigmoid())
    
    def forward(self, data):
        initial_x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Начальная обработка рёбер для получения их скрытых представлений
        row, col = edge_index
        initial_edge_features = torch.cat([initial_x[row], initial_x[col], edge_attr], dim=1)
        edge_hidden = self.edge_encoder(initial_edge_features)
        
        # Начальная обработка узлов для получения их скрытых представлений
        x = self.node_encoder(initial_x)

        # Начальное получение меток рёбер
        edge_features = torch.cat([x[row], x[col], edge_hidden, initial_x[row], initial_x[col], edge_attr], dim=1)
        edge_labels = self.initial_edge_classification_mlp(edge_features)

        for _ in range(self.n_iters):
            # Обновление скрытых представлений узлов
            x = self.node_gatconv(x, edge_index, edge_hidden, edge_labels.squeeze(), initial_x)

            # Обновление скрытых представлений рёбер
            edge_features = torch.cat([x[row], x[col], edge_hidden, edge_labels, initial_x[row], initial_x[col], edge_attr], dim=1)
            edge_hidden = self.edge_mlp(edge_features)

            # Обновление меток рёбер
            edge_features = torch.cat([x[row], x[col], edge_hidden, edge_labels, initial_x[row], initial_x[col], edge_attr], dim=1)
            edge_labels = self.edge_classification_mlp(edge_features)
        
        return edge_labels.squeeze()

    
# Train function   
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        edge_logits = model(data)
        loss = criterion(edge_logits, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            edge_logits = model(data)
            all_true_labels.append(data.y.cpu().numpy())
            all_pred_labels.append((edge_logits >= threshold).cpu().numpy())
    
    # Evaluate metrics
    all_true_labels = np.concatenate(all_true_labels)
    all_pred_labels = np.concatenate(all_pred_labels)
    
    true_positive = np.sum((all_pred_labels == 1) & (all_true_labels == 1))
    true_negative = np.sum((all_pred_labels == 0) & (all_true_labels == 0))
    false_positive = np.sum((all_pred_labels == 1) & (all_true_labels == 0))
    false_negative = np.sum((all_pred_labels == 0) & (all_true_labels == 1))
    
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    purity = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    efficiency = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    
    return accuracy, purity, efficiency

def balanced_cross_entropy(pred, target, pos_weight, neg_weight):
    # Calculate the weights for positive and negative samples
    weights = target * pos_weight + (1 - target) * neg_weight
    
    # Compute the binary cross-entropy loss
    loss = F.binary_cross_entropy(pred, target, weight=weights, reduction='mean')
    
    return loss

def main():
    start_time = time.time()

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
    epoch_num = parameters['epoch_num']
    batch_size = parameters['batch_size']
    n_iters = parameters['n_iters']
    learning_rate = parameters['learning_rate']
    step_size = parameters['step_size']
    gamma = parameters['gamma']
    pos_weight = parameters['pos_weight']
    neg_weight = parameters['neg_weight']
    time_lapse = parameters['time_lapse']

    # Load data
    graph_files = [os.path.join(input_dir, f'event_{evtid}_section_{section_id}_graph.npz') for evtid, section_id in product(range(n_files), range(section_num))]
    dataset = [load_npz_to_pyg(file) for file in graph_files]

    # Split data into train and test sets
    train_len = int(len(dataset) * (1-test_size))
    train_dataset = dataset[:train_len]
    test_dataset = dataset[train_len:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create log file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.basicConfig(filename=os.path.join(output_dir, 'model_metrics.log'), level=logging.INFO)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device, "device")

    # Initializing the model and optimizer
    node_feature_dim = dataset[0].x.size(1)
    edge_feature_dim = dataset[0].edge_attr.size(1)
    model = EdgeClassificationGNN(node_feature_dim, edge_feature_dim, n_iters).to(device) 
    optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(device)
    neg_weight = torch.tensor(neg_weight, dtype=torch.float).to(device)
    criterion = lambda pred, target: balanced_cross_entropy(pred, target, pos_weight, neg_weight)
    
    # Watch the time and set the folder to save the model
    last_save_time = time.time()
    model_save_path = os.path.join(output_dir, 'model_checkpoint.pth')

    # Check for existing checkpoint
    start_epoch = 0
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, weights_only=True, map_location=device)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'epoch' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f'Loaded checkpoint from epoch {start_epoch}')
        else:
            print('Checkpoint file is missing some keys, starting from scratch.')

    # Training the model
    for epoch in range(start_epoch, epoch_num):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        accuracy, purity, efficiency = evaluate(model, test_loader, device)
        status = f'Epoch {epoch+1}, Train Loss: {train_loss:.5f}, Accuracy: {accuracy:.4f}, Purity: {purity:.4f}, Efficiency: {efficiency:.4f}'
        logging.info(status)
        current_time = time.time()
        if current_time - last_save_time >= time_lapse or epoch == epoch_num-1:
            if os.path.exists(model_save_path):
                os.remove(model_save_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, model_save_path)
            last_save_time = current_time
            print('Model saved: ' + status)
    
    print(f"Spent time: {time.time() - start_time:.3f} s")


if __name__ == '__main__':
    main()