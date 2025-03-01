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


def select_hits_for_training(hits, truth, tracks):
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    truth = truth.merge(tracks[['track_id', 'pt']], on='track_id')
    hits = hits[['hit_id', 'z', 'row_id']].assign(r=r, phi=phi).merge(truth, on='hit_id')

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

def select_segments(hits1, hits2, dphi_max, z0_max, dtheta_max, d_min, d_max):
    # Start with all possible pairs of hits
    keys = ['evtid', 'r', 'phi', 'z']
    hit_pairs = hits1[keys].reset_index().merge(hits2[keys].reset_index(), on='evtid', suffixes=('_1', '_2'))

    # Compute line through the points
    dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
    dz = hit_pairs.z_2 - hit_pairs.z_1
    dr = hit_pairs.r_2 - hit_pairs.r_1
    z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr
    distance = np.sqrt(hit_pairs.r_1**2 + hit_pairs.r_2**2 - 2 * hit_pairs.r_1 * hit_pairs.r_2 * np.cos(dphi) + dz**2)
    dtheta = np.arctan(dz / dr)

    # Filter segments according to criteria
    good_seg_mask = (
        (dphi.abs() < dphi_max) &
        (z0.abs() < z0_max) &
        (dtheta.abs() < dtheta_max) &
        (distance > d_min) & (distance < d_max)
    )

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

def process_func(evtid, input_dir, output_dir, phi_edges, eta_edges, num_rows, node_feature_scale, dphi_max, z0_max, dtheta_max, d_min, d_max, pt_min):
        # The name of the current file 
        event_name = input_dir + f'/event_{evtid}_'
        print(f'Processing: {event_name:s}hits/truth.csv')

        # Read hits and truth files
        hits_df = pd.read_csv(event_name + 'hits.csv')
        truth_df = pd.read_csv(event_name + 'truth.csv')
        tracks_df = pd.read_csv(event_name + 'tracks.csv')

        # Select only necessary hits data
        hits_df = select_hits_for_training(hits_df, truth_df, tracks_df).assign(evtid=evtid)

        # Split all hits into detectors sections
        hits_sections = split_detector_sections(hits_df, phi_edges, eta_edges)

        # Define adjacent rows
        l = np.arange(num_rows)
        row_pairs = np.stack([l[:-1], l[1:]], axis=1)

        filtered_track_segments = dict()

        # Loop over row pairs and construct segments
        for section_id, hits in enumerate(hits_sections):
            # Make graph
            G = nx.DiGraph()
            for idx, row in hits.iterrows():
                G.add_node(idx, pos=row[['r', 'phi', 'z']].to_numpy() / node_feature_scale)

            # Take row groups
            row_groups = hits.groupby('row_id')
            segments = []
            for (row1, row2) in row_pairs:
                # Find and join all hit pairs
                try:
                    hits1 = row_groups.get_group(row1)
                    hits2 = row_groups.get_group(row2)

                # If an event has no hits on a row, we get a KeyError.
                except KeyError as exc:
                    logging.info(f'Skipping empty row: {exc}')
                    continue

                # Construct the segments
                segments.append(select_segments(hits1, hits2, dphi_max, z0_max, dtheta_max, d_min, d_max))

            # Combine segments from all row pairs
            segments = pd.concat(segments)

            # Add edges to the graph
            for idx, row in segments.iterrows():
                index1 = row['index_1']
                index2 = row['index_2']
                edge_lable = 0
                filtered_hits1 = hits[(hits['track_id'] == hits.loc[index1, 'track_id']) & (hits['row_id'] == hits.loc[index1, 'row_id'])]
                filtered_hits2 = hits[(hits['track_id'] == hits.loc[index2, 'track_id']) & (hits['row_id'] == hits.loc[index2, 'row_id'])]
                if hits.loc[index1, 'track_id'] == hits.loc[index2, 'track_id'] and hits.loc[index1, 'pt'] >= pt_min and \
                index1 == filtered_hits1['phi'].idxmin() and index2 == filtered_hits2['phi'].idxmin():
                    edge_lable = 1
                    filtered_track_segments.setdefault(hits.loc[index1, 'track_id'], set()).add((hits.loc[index1, 'row_id'], hits.loc[index2, 'row_id']))
                G.add_edge(index1, index2, features=get_edge_features(G.nodes[index1]['pos'], G.nodes[index2]['pos']), label=edge_lable)

            # Save graph
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_graph_to_npz(G, output_dir + f'/event_{evtid}_section_{section_id}_graph.npz')

        filtered_track_id_groups = hits_df[hits_df['pt'] >= pt_min].groupby('track_id')

        av_eff = 0
        for key in filtered_track_segments:
            numbers = set(filtered_track_id_groups.get_group(key)['row_id'])
            count = 0
            for num in numbers:
                if num + 1 in numbers:
                    count += 1
            av_eff += len(filtered_track_segments[key]) / (count + 1e-8)

        av_eff /= len(filtered_track_segments.keys()) + 1e-8
        
        print(f'Integrity: {av_eff * 100:.2f}')
        
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
    dphi_max = selection['dphi_max']
    z0_max = selection['z0_max']
    dtheta_max = selection['dtheta_max']
    d_min = selection['d_min']
    d_max = selection['d_max']
    pt_min = selection['pt_min']
    n_phi_sections = selection['n_phi_sections']
    n_eta_sections = selection['n_eta_sections']
    eta_range = selection['eta_range']
    num_rows = selection['num_rows']
    rmax = selection['rmax']
    zmax = selection['zmax']
    
    phi_range = [-np.pi, np.pi]
    phi_edges = np.linspace(*phi_range, num=n_phi_sections+1)
    eta_edges = np.linspace(*eta_range, num=n_eta_sections+1)

    # Parameters of feature nodes normalization
    node_feature_scale = np.array([rmax, np.pi, zmax])

    # Process input files with a worker pool
    with mp.Pool(processes=args.j) as pool: 
        partial_func = partial(process_func, input_dir=input_dir, output_dir=output_dir, 
                               phi_edges=phi_edges, eta_edges=eta_edges, num_rows=num_rows,
                               node_feature_scale=node_feature_scale, 
                               dphi_max=dphi_max, z0_max=z0_max,
                               dtheta_max=dtheta_max, d_min=d_min, d_max=d_max, pt_min=pt_min)
        pool.map(partial_func, range(n_files))


if __name__ == '__main__':
    main()