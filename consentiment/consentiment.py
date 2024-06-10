
import pandas as pd
import numpy as np
import networkx as nx
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from typing import List

def create_graph_from_df(df:pd.DataFrame, text:str='text', ignore_model:List[str]=None, embedding_mi:float=0.9, distance_threshold:float=0.1, verbose:bool=True):
    G = nx.Graph()

    for col in df.columns:
        if col == text or col == 'annotation' or col.endswith("_emb"): continue
        if ignore_model is not None:
            if type(ignore_model) is not list:
                ignore_model = [ignore_model]
            to_skip = False
            for igm in ignore_model:
                if igm in col:
                    if(verbose): print("Jumping", col)
                    to_skip = True
                    break
            if to_skip: continue

        if(verbose): print("Adding", col, "to graph.")

        for i, row in df.iterrows():
            annotation = np.asarray(df.iloc[i]['annotation'])
            if len(annotation) == 0: continue

            annotation_aspects = annotation[:, 0]
            for j, (aspect, sentiment) in enumerate(row[col]):
                node_id = f"{i}_{col}_{aspect}_{sentiment}"
                aspect_embedding = embedding_mi * np.array(row[col+'_ents_emb'][j]) + (1.0-embedding_mi) * np.array(row[text+'_emb'])

                annotation_i = np.where(annotation_aspects == aspect)[0]
                if len(annotation_i) >= 1:
                    annotation_sentiment = annotation[annotation_i, 1][0]
                else:
                    annotation_sentiment = 'NONE'
                    continue

                node_data = {
                    'text': row[text],
                    'model': col,
                    'aspect': aspect,
                    'sentiment': sentiment,
                    'annotation': annotation_sentiment,
                    'embedding': aspect_embedding
                }
                G.add_node(node_id, **node_data)


    for node1, data1 in tqdm(G.nodes(data=True)):
        for node2, data2 in G.nodes(data=True):
            # Linking same aspects with different sentiments
            if node1 != node2 and data1['aspect'].lower() == data2['aspect'].lower() and data1['sentiment'] != data2['sentiment']:
                weight = sum([a*b for a,b in zip(data1['embedding'], data2['embedding'])])
                G.add_edge(node1, node2, weight=weight)
            # Linking close embeddings with same sentiments
            elif node1 != node2 and data1['sentiment'] == data2['sentiment']:
                # dist = distance.cosine(data1['embedding'], data2['embedding'])
                dist = 1.0 - np.dot(data1['embedding'], data2['embedding']) / (np.linalg.norm(data1['embedding']) * np.linalg.norm(data2['embedding']))
                if dist < distance_threshold:
                    G.add_edge(node1, node2, weight=dist)
    return G

from tqdm.notebook import tqdm
import random

def regularize(G, num_labels=3, iterations=10, mi=0.1):

    nodes = []

    # inicializando vetor f para todos os nodes
    for node in G.nodes():
        if 'f' not in G.nodes[node]:
            G.nodes[node]['f'] = np.array([0.0]*num_labels)
            G.nodes[node]['y'] = np.array([0.0]*num_labels)
            if G.nodes[node]['sentiment']   == 'NEG':
                G.nodes[node]['y'][0] = 1.0
                G.nodes[node]['f'][0] = 1.0
            elif G.nodes[node]['sentiment'] == 'NEU':
                G.nodes[node]['y'][1] = 1.0
                G.nodes[node]['f'][1] = 1.0
            elif G.nodes[node]['sentiment'] == 'POS':
                G.nodes[node]['y'][2] = 1.0
                G.nodes[node]['f'][2] = 1.0
        nodes.append(node)

    pbar = tqdm(range(0, iterations))

    for iteration in pbar:
        random.shuffle(nodes)
        energy = 0.0

        for node in nodes:
            f_new = np.array([0.0]*num_labels)
            f_old = np.array(G.nodes[node]['f'])*1.0
            sum_w = 0.0

            for neighbor in G.neighbors(node):

                  w = 0.0
                  if 'weight' in G[node][neighbor]:
                      w = G[node][neighbor]['weight']

                  f_new += w*G.nodes[neighbor]['f']

                  sum_w += w

            if sum_w!=0.0:
              f_new /= sum_w
            else:
              f_new = f_old

            G.nodes[node]['f'] = f_new*1.0

            G.nodes[node]['f'] = G.nodes[node]['y'] * mi + G.nodes[node]['f']*(1.0-mi)

            energy += np.linalg.norm(f_new-f_old)

        iteration += 1

        message = 'Iteration '+str(iteration)+' | Energy = '+str(energy)
        # print(message)
        pbar.set_description(message)

    return G

def reset_regularization(G):
    for node in G.nodes():
        if 'f' not in G.nodes[node]: return
        break

    for node in G.nodes():
        G.nodes[node]['f'] = G.nodes[node]['y']

def graph_committee_to_data(G, bdf, text:str="text"):
    isent_to_sent = {0:'NEG', 1:'NEU', 2:'POS'}

    df = bdf[[text, 'annotation']].copy()

    id_aspect_dict = [{} for _ in range(len(df))]

    for node in G.nodes():
        id, model, aspect, sentiment = node.split('_')
        id = int(id)

        if aspect not in id_aspect_dict[id]:
            id_aspect_dict[id][aspect] = []
        id_aspect_dict[id][aspect].append(G.nodes[node]['f'])

    data = []
    for id in range(len(df)):
        data.append([])
        for aspect in id_aspect_dict[id]:
            new_sent = isent_to_sent[np.argmax(np.mean(id_aspect_dict[id][aspect], axis=0))]
            data[-1].append((aspect, new_sent))
    return data
