#!/usr/bin/env python

import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_graph_from_npz(filename):
    data = np.load(filename, allow_pickle=True)
    
    G = nx.Graph()
    
    for i, node in enumerate(data['nodes']):
        G.add_node(node, pos=data['node_pos'][i])
    
    for edge, features, label in zip(data['edges'], data['edge_features'], data['edge_labels']):
        G.add_edge(*edge, features=features, label=label)
    
    return G

def get_pos(Gp):
    pos = {}
    for node in Gp.nodes():
        r, phi, z = Gp.nodes[node]['pos'][:3]
        x = r * np.cos(np.pi * phi)
        y = r * np.sin(np.pi * phi)
        pos[node] = np.array([x, y])
    return pos

def get_3Dpos(Gp):
    pos = {}
    for node in Gp.nodes():
        r, phi, z = Gp.nodes[node]['pos'][:3]
        x = r * np.cos(np.pi * phi)
        y = r * np.sin(np.pi * phi)
        pos[node] = np.array([x, y, z])
    return pos

def plot_graph(G):
    n_edges = len(G.edges())
    edge_colors = ['green']*n_edges
    edge_alpha = [1.]*n_edges
    for iedge,edge in enumerate(G.edges(data=True)):
        if int(edge[2]['label']) != 1:
            edge_colors[iedge] = 'grey'
            edge_alpha[iedge] = 0.3
    pos = get_pos(G) 
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    widths = [2.0 if color == 'r' else 1.5 for color in edge_colors]

    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=3, alpha=1.0, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=widths, alpha=edge_alpha, ax=ax, arrows=False)

    plt.show()

def plot3D_networkx(G, animate=False, only_true=False):
    
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
        linewidth = 2
        if int(edge[2]['label']) != 1:
            col = 'grey'
            alpha = 0.3
            linewidth = 0.7
        if int(edge[2]['label']) == 1:
            ax.plot([p1[0] * np.cos(np.pi * p1[1]), p2[0] * np.cos(np.pi * p2[1])], 
                    [p1[0] * np.sin(np.pi * p1[1]), p2[0] * np.sin(np.pi * p2[1])],
                    [p1[2], p2[2]], color=col, linewidth=linewidth, alpha=alpha)
        elif not only_true:
            ax.plot([p1[0] * np.cos(np.pi * p1[1]), p2[0] * np.cos(np.pi * p2[1])], 
                    [p1[0] * np.sin(np.pi * p1[1]), p2[0] * np.sin(np.pi * p2[1])],
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

        angle = np.linspace(0, 360, 360)
        ani = FuncAnimation(fig, animate, angle, interval=150, blit=True)

        ani.save('3D_graph_animation.gif', writer=PillowWriter(fps=30))

def parse_args():
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('graph_file')
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
    # Plot graph
    plot_graph(G)
    # Plot 3D
    plot3D_networkx(G, str_to_bool(args.animate), str_to_bool(args.only_true))

if __name__ == '__main__':
    main()