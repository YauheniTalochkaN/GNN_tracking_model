#!/usr/bin/env python

import argparse
import logging
import yaml
import os
import networkx as nx
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial


def select_hits_for_training(hits, truth, remove_noise_fraction=1):
    # Barrel volume and layer ids
    vlids = [(8,2), (8,4), (8,6), (8,8),
             (13,2), (13,4), (13,6), (13,8),
             (17,2), (17,4)]
    n_det_layers = len(vlids)

    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i) for i in range(n_det_layers)])

    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)

    # Select the data columns we need
    hits = (hits[['hit_id', 'z', 'layer']].assign(r=r, phi=phi).merge(truth[['hit_id', 'particle_id']], on='hit_id'))

    # Take noise
    noise = hits[hits['particle_id'] == 0].copy()
    noise['particle_id'] = range(-1, -len(noise['particle_id']) - 1, -1)

    # Remove some part of noise hits
    noise = noise.sample(n=int(len(noise) * (1-remove_noise_fraction)))

    # Remove noise
    hits = hits[hits['particle_id'] != 0]

    # Remove duplicate hits
    hits = hits.loc[hits.groupby(['particle_id', 'layer'])['r'].idxmin()]

    # Contac noise and particle hits
    hits = pd.concat([hits, noise], ignore_index=True)
    """ print("Positive hits: ", len(hits[hits['particle_id'] > 0]))
    print("Negative hits: ", len(hits[hits['particle_id'] < 0]))
    print("Noise level: ", len(hits[hits['particle_id'] < 0]) / len(hits[hits['particle_id'] > 0])) """

    return hits

def split_detector_sections(hits, phi_edges, eta_edges):
    hits_sections = []

    # Loop over sections
    for i in range(len(phi_edges) - 1):
        phi_min, phi_max = phi_edges[i], phi_edges[i+1]

        # Select hits in this phi section
        phi_hits = hits[(hits.phi > phi_min) & (hits.phi < phi_max)]

        # Center these hits on phi=0
        centered_phi = phi_hits.phi - (phi_min + phi_max) / 2
        phi_hits = phi_hits.assign(phi=centered_phi, phi_section=i)
        for j in range(len(eta_edges) - 1):
            eta_min, eta_max = eta_edges[j], eta_edges[j+1]
            # Select hits in this eta section
            eta = calc_eta(phi_hits.r, phi_hits.z)
            sec_hits = phi_hits[(eta > eta_min) & (eta < eta_max)]
            hits_sections.append(sec_hits.assign(eta_section=j))

    return hits_sections

def calc_dphi(phi1, phi2):
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def calc_dphi_2(phi1, phi2):
    dphi = phi2 - phi1
    if dphi > np.pi:
        dphi -= 2*np.pi
    if dphi < -np.pi:
        dphi += 2*np.pi
    return dphi

def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))

def select_segments(hits1, hits2, phi_slope_max, z0_max):
    # Start with all possible pairs of hits
    keys = ['evtid', 'r', 'phi', 'z']
    hit_pairs = hits1[keys].reset_index().merge(hits2[keys].reset_index(), on='evtid', suffixes=('_1', '_2'))

    # Compute line through the points
    dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
    dz = hit_pairs.z_2 - hit_pairs.z_1
    dr = hit_pairs.r_2 - hit_pairs.r_1
    phi_slope = dphi / dr
    z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr

    # Filter segments according to criteria
    good_seg_mask = (phi_slope.abs() < phi_slope_max) & (z0.abs() < z0_max)

    return hit_pairs[['index_1', 'index_2']][good_seg_mask]

def get_pos(Gp):
    pos = {}
    for node in Gp.nodes():
        r, phi, z = Gp.nodes[node]['pos'][:3]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        pos[node] = np.array([x, y])
    return pos

def get_edge_features(in_node, out_node):
    # Calculate r, phi and z of input and output nodes
    in_r, in_phi, in_z    = in_node
    out_r, out_phi, out_z = out_node

    # Evaluate spherical radius of input and output nodes
    in_r3 = np.sqrt(in_r**2 + in_z**2)
    out_r3 = np.sqrt(out_r**2 + out_z**2)

    # Evaluate theta and eta coordinates of input and output nodes
    in_theta = np.arccos(in_z/in_r3)
    in_eta = -np.log(np.tan(in_theta/2.0))
    out_theta = np.arccos(out_z/out_r3)
    out_eta = -np.log(np.tan(out_theta/2.0))

    # Calculate edge features
    deta = out_eta - in_eta
    dphi = calc_dphi_2(out_phi, in_phi)
    dR = np.sqrt(deta**2 + dphi**2)
    dZ = in_z - out_z

    return np.array([deta, dphi, dR, dZ])

def save_graph_to_npz(G, filename):
    #Set default node numbers
    mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # Get graph nodes
    nodes = list(G.nodes())
    
    # Get node positions
    node_pos = np.array([G.nodes[n]['pos'] for n in nodes])
    
    # Get edge features and labels
    edge_features = []
    edge_labels = []
    edges = []
    for e in G.edges():
        edges.append(e)
        edge_features.append(G.edges[e]['features'])
        edge_labels.append(G.edges[e]['label'])
    
    # Convert all lists into numpy arrays
    edges = np.array(edges)
    edge_features = np.array(edge_features)
    edge_labels = np.array(edge_labels)
    
    # Save data as npz
    np.savez(filename,
             nodes=nodes,
             node_pos=node_pos,
             edges=edges,
             edge_features=edge_features,
             edge_labels=edge_labels)

def parse_args():
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/preprocessing_parameters.yaml')
    add_arg('--j', type=int, default=1)
    return parser.parse_args()

def process_func(evtid, input_dir, output_dir, phi_edges, eta_edges, node_feature_names, node_feature_scale, phi_slope_max, z0_max, remove_noise_fraction):
        # The name of the current file 
        event_name = input_dir + f'/event0000010{evtid:02}-'
        print(f'Processing: {event_name:s}hits/truth.csv')

        # Read hits and truth files
        hits_df = pd.read_csv(event_name + 'hits.csv')
        truth_df = pd.read_csv(event_name + 'truth.csv')

        # Select only necessary hits data
        hits_df = select_hits_for_training(hits_df, truth_df, remove_noise_fraction).assign(evtid=evtid)

        # Split all hits into detectors sections
        hits_sections = split_detector_sections(hits_df, phi_edges, eta_edges)

        # Define adjacent layers
        n_det_layers = 10
        l = np.arange(n_det_layers)
        layer_pairs = np.stack([l[:-1], l[1:]], axis=1)

        # Loop over layer pairs and construct segments
        for section_id, hits in enumerate(hits_sections):
            # Make graph
            G = nx.Graph()
            for idx, row in hits.iterrows():
                G.add_node(idx, pos=row[node_feature_names].to_numpy() / node_feature_scale)

            # Take layer groups
            layer_groups = hits.groupby('layer')
            segments = []
            for (layer1, layer2) in layer_pairs:
                # Find and join all hit pairs
                try:
                    hits1 = layer_groups.get_group(layer1)
                    hits2 = layer_groups.get_group(layer2)

                # If an event has no hits on a layer, we get a KeyError.
                except KeyError as exc:
                    logging.info(f'Skipping empty layer: {exc:s}')
                    continue

                # Construct the segments
                segments.append(select_segments(hits1, hits2, phi_slope_max, z0_max))

            # Combine segments from all layer pairs
            segments = pd.concat(segments)

            # Add edges to the graph
            for _, row in segments.iterrows():
                index1 = row['index_1']
                index2 = row['index_2']
                edge_lable = 1
                if hits.loc[index1, 'particle_id'] != hits.loc[index2, 'particle_id']:
                    edge_lable = 0
                G.add_edge(index1, index2, features=get_edge_features(G.nodes[index1]['pos'], G.nodes[index2]['pos']), label=edge_lable)

            # Save graph
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_graph_to_npz(G, output_dir + f'/event0000010{evtid:02}_section{section_id:02}_graph.npz')

def main():
    # Get args
    args = parse_args()

    # Open the file of preprocessing parameters
    with open(args.config, 'r') as f:
        parameters = yaml.safe_load(f)

    # Obtain necessary parameters
    input_dir = parameters['input_dir']
    output_dir = parameters['output_dir']
    n_files = parameters['n_files']
    selection = parameters['selection']
    phi_slope_max = selection['phi_slope_max']
    z0_max = selection['z0_max']
    n_phi_sections = selection['n_phi_sections']
    n_eta_sections = selection['n_eta_sections']
    eta_range = selection['eta_range']
    remove_noise_fraction = selection['remove_noise_fraction']
    
    phi_range = [-np.pi, np.pi]
    phi_edges = np.linspace(*phi_range, num=n_phi_sections+1)
    eta_edges = np.linspace(*eta_range, num=n_eta_sections+1)

    # Parameters of feature nodes normalization
    node_feature_names = ['r', 'phi', 'z']
    node_feature_scale = np.array([1000., np.pi / n_phi_sections, 1000.])

    # Process input files with a worker pool
    with mp.Pool(processes=args.j) as pool: 
        partial_func = partial(process_func, input_dir=input_dir, output_dir=output_dir, 
                               phi_edges=phi_edges, eta_edges=eta_edges, 
                               node_feature_names=node_feature_names, node_feature_scale=node_feature_scale, 
                               phi_slope_max=phi_slope_max, z0_max=z0_max, remove_noise_fraction=remove_noise_fraction)
        pool.map(partial_func, range(n_files))


if __name__ == '__main__':
    main()