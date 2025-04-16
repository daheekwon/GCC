import glob
import os
import pickle
from collections import defaultdict


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from tqdm.auto import tqdm

LAYERS = ["layer1.0", "layer1.1", "layer1.2", 
          "layer2.0", "layer2.1", "layer2.2", "layer2.3",
          "layer3.0", "layer3.1", "layer3.2", "layer3.3", "layer3.4", "layer3.5",
          "layer4.0", "layer4.1", "layer4.2"]

def create_graph_from_paths(paths):
    G = nx.DiGraph()
    for path in paths:
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i+1])
    return G

def create_graph_from_nodes(nodes):
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node, layer=float(node[0].replace('layer', '')), cid=node[1])
    return G

def add_node_attributes_layer_cid(graph):
    for node in graph.nodes:
        layer = node[0]
        cid = node[1]
        print(node)
        if isinstance(layer, str):
            graph.nodes[node]['layer'] = float(layer.replace('layer', ''))
        else:
            graph.nodes[node]['layer'] = layer
        graph.nodes[node]['cid'] = cid
    return graph
    
def add_edge_attributes_weight(graph, ref_graph):
    for edge in graph.edges:
        if edge in ref_graph.edges:
            graph.edges[edge]['weight'] = ref_graph.edges[edge]['weight']
        else:
            print(f"edge {edge} not in ref_graph")
    return graph

def convert_matrices_to_connections(matrices, layer_names=LAYERS):
    """
    Convert a sparse matrix to a dictionary of connections.
    """
    connections = defaultdict(list)
    for i, matrix in enumerate(matrices):
        layer_name = layer_names[i]
        rows, cols = matrix.nonzero()
        values = matrix.data
        for src_ch, tgt_ch, weight in zip(rows, cols, values):
            # print(layer_name, src_ch, tgt_ch, weight)
            connections[(layer_name, src_ch)].append((layer_names[i+1], tgt_ch, weight))

    return connections


def visualize_from_pkl(pkl_path, img_dir, img_size=300, is_save=True, width=None, height=None, option='cropped_image', pos=None):
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    start_layer = results['src_layer_block'].replace('_block', '.') # layer3_block1
    loc = LAYERS.index(start_layer)
    layer_names = LAYERS[loc:loc+len(results['matrices'])+1]
    
    try:
        connections = convert_matrices_to_connections(results['matrices'], layer_names)
        print(connections)
        
        # Check empty connections
        if not connections:
            print(pkl_path)
            print("Connections dictionary is empty")
            return
        
        G = create_graph_from_connections(connections, img_dir, img_size=img_size, option=option)
        if pos is None:
            pos = compute_positions(G)
        
        max_connections_length = len(max(connections.values(), key=len))
        num_layers = len(set([key[0] for key in connections.keys()])) + 1
        print(num_layers, max_connections_length)
        if max_connections_length <= 1:
            return
            
        if width is None:
            width = 0.7 * num_layers
        if height is None:
            height = 0.3 * max_connections_length
        dpi = 100
        fig = plt.figure(figsize=(width, height), dpi=dpi)

        nx.draw(G, pos, 
            with_labels=False,
            width=[max(0.1, e[2]['width'] * 2) for e in G.edges(data=True)])

        min_y = 50000
        x_list = []
        ax = plt.gca()

        for node in G.nodes():
            img = G.nodes[node]['image']
            img = OffsetImage(img, zoom=0.2)  # Adjust zoom as needed
            ab = AnnotationBbox(img, pos[node], frameon=False)
            ax.add_artist(ab)

            # Add text to nodes (channel number)
            text = G.nodes[node]['text']
            x, y = pos[node]
            ax.text(
                x, y - np.abs(y) * 0.2, text, ha='center', va='top', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgrey', boxstyle='round,pad=0.5')
            )

            # Get lowest x position
            if y < min_y:
                min_y = y
            if x not in x_list:
                x_list.append(x)

        # Add layer name annotation
        unique_layer_names = sorted(set(node[0] for node in G.nodes()))
        assert len(unique_layer_names) == len(x_list)

        for layer_name, x in zip(unique_layer_names, x_list):
            ax.text(x, min_y - np.abs(min_y) * 0.5, layer_name, ha='center', va='center', fontsize=8)

        fname = os.path.basename(pkl_path).replace('connection_matrices_from', 'graph')
        fname = fname.replace('.pkl', '.png')
        
        fig.tight_layout()
        if is_save:
            plt.savefig(os.path.join(os.path.dirname(pkl_path), fname))
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(pkl_path)
        print(e)
        print(layer_names)
        print(connections)
        raise e

FEATURE_DIMS = {
    'layer1.0': 256,
    'layer1.1': 256,
    'layer1.2': 256,

    'layer2.0': 512,
    'layer2.1': 512,
    'layer2.2': 512,
    'layer2.3': 512,

    'layer3.0': 1024,
    'layer3.1': 1024,
    'layer3.2': 1024,
    'layer3.3': 1024,
    'layer3.4': 1024,
    'layer3.5': 1024,

    'layer4.0': 2048,
    'layer4.1': 2048,
    'layer4.2': 2048,
}

LAYER_POS = {
    'layer1.0': 0,
    'layer1.1': 1,
    'layer1.2': 2,
    
    'layer2.0': 3,
    'layer2.1': 4,
    'layer2.2': 5,
    'layer2.3': 6,

    'layer3.0': 7,
    'layer3.1': 8,
    'layer3.2': 9,
    'layer3.3': 10,
    'layer3.4': 11,
    'layer3.5': 12,

    'layer4.0': 13,
    'layer4.1': 14,
    'layer4.2': 15,
}

def compute_absolute_positions(G):
    pos = {}
    for node in G.nodes():
        layer_name = f'layer{G.nodes[node]["layer"]}'
        cid = G.nodes[node]['cid']

        num_channels = FEATURE_DIMS[layer_name]
        x = LAYER_POS[layer_name]
        y = cid
        pos[node] = (x, y)
    return pos

def compute_positions(G):
    """
    Compute positions for the nodes in the graph G.
    
    Args:
        G: A NetworkX directed graph.
    
    Returns:
        pos: A dictionary with node names as keys and their positions as values.
    """
    pos = {}
    layer_nodes = {}
    
    # Extract layer information from the graph
    for node in G.nodes():
        layer_idx = G.nodes[node]['layer']
        if isinstance(layer_idx, str):
            if 'layer' in layer_idx:
                layer_idx = layer_idx.replace('layer', '')
            layer_idx = float(layer_idx)
        if layer_idx not in layer_nodes:
            layer_nodes[layer_idx] = []
        layer_nodes[layer_idx].append(node)
    
    layers = sorted(layer_nodes.keys())
    # Compute positions
    for i, layer_idx in enumerate(layers):
        n_nodes = len(layer_nodes[layer_idx])
        for node_idx, node_name in enumerate(layer_nodes[layer_idx]):
            layer_idx_int = layers.index(layer_idx)
            # pos[node_name] = (i * 1, node_idx - n_nodes / 2)  # Arrange nodes vertically
            pos[node_name] = (layer_idx_int * 3, node_idx - n_nodes/2)  # Arrange nodes vertically
    
    return pos

def load_image(img_path, img_size=300):
    if os.path.exists(img_path):
        img = Image.open(img_path)
        if isinstance(img_size, tuple):
            img = img.resize((img_size[0], img_size[1]))
        else:
            img = img.resize((img_size, img_size))
    else:
        print(f"Does not exist: {img_path}")
        img = np.zeros((img_size, img_size, 3))
    return img

def visualize_path(path, option='cropped_image', img_size=300):
    nrows = 1
    ncols = len(path)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    for i, node in enumerate(path):
        layer_name = node[0]
        cid = node[1]
        
        img_dir = '/project/PURE/results/global_features/imagenet/resnet50_torchvision'
    
        img = load_image(os.path.join(img_dir, layer_name, option, f'{cid:04d}.png'), img_size)
        axes[i].set_title(f'{layer_name} Ch{cid}')
        axes[i].imshow(img)
        axes[i].set_xticks([])
        axes[i].set_yticks([])  

    plt.tight_layout()
    plt.show()

def create_graph_from_connections(connections, img_dir=None, option='cropped_image', img_size=300):
    """
    Create a graph from a dictionary of connections.
    connections = {('layer1.0', 17): [('layer1.1', 17, 0.938), ('layer1.1', 95, 0.236)],
                   ('layer1.1', 17): [('layer1.2', 17, 0.868)], 
                   ('layer1.2', 17): [('layer2.0', 202, 0.17), ('layer2.0', 352, 0.134)]}
    """
    # Create the graph
    G = nx.DiGraph()

    # Add nodes and edges
    for source, targets in connections.items():
        layer_name, cid = source
        if 'layer' in layer_name and 'block' in layer_name:
            layer_id = float(layer_name.replace('layer', '').replace('_block', '.'))
        else:
            layer_id = float(layer_name.replace('layer', ''))
        G.add_node(source, layer=layer_id, cid=cid, text=f'Ch{cid}')

        for target_layer, target_node, weight in targets:
            target = (target_layer, target_node)
            if 'layer' in layer_name and 'block' in layer_name:
                layer_id = float(layer_name.replace('layer', '').replace('_block', '.'))
            else:
                layer_id = float(layer_name.replace('layer', ''))

            G.add_node(target, layer=layer_id, cid=target_node, text=f'Ch{target_node}')
            G.add_edge(source, target, weight=weight)

    # Add image attributes to nodes
    if img_dir is not None:
        for node in G.nodes():
            img_path = os.path.join(img_dir, G.nodes[node]['layer'], option, f"{G.nodes[node]['cid']:04d}.png")
            img = load_image(img_path, img_size)
            G.nodes[node]['image'] = img

    return G


def create_connection_graph(matrices, threshold=0.1):
    """
    Create a graph visualization of channel connections
    Args:
        matrices: List of sparse matrices
        threshold: Minimum absolute weight to show connection
    """
    G = nx.DiGraph()
    
    # Create position dictionary for layers
    pos = {}
    layer_nodes = {}
    
    # Add nodes for each layer
    for layer_idx in range(len(matrices) + 1):
        if layer_idx == 0:
            matrix = matrices[0]
            n_nodes = matrix.shape[0]
        elif layer_idx == len(matrices):
            matrix = matrices[-1]
            n_nodes = matrix.shape[1]
        else:
            matrix = matrices[layer_idx]
            n_nodes = matrix.shape[0]
            
        layer_nodes[layer_idx] = []
        for node_idx in range(n_nodes):
            node_name = f'L{layer_idx}_Ch{node_idx}'
            G.add_node(node_name)
            pos[node_name] = (layer_idx * 3, node_idx - n_nodes/2)  # Arrange nodes vertically
            layer_nodes[layer_idx].append(node_name)

        # 빈 레이어는 skip
        if layer_idx < len(matrices):
            rows, cols = matrices[layer_idx].nonzero()
            if len(rows) == 0:
                break
    
    # Add edges from matrices
    for layer_idx, matrix in enumerate(matrices):
        rows, cols = matrix.nonzero()
        values = matrix.data
        
        for row, col, val in zip(rows, cols, values):
            if abs(val) > threshold:
                src_node = f'L{layer_idx}_Ch{row}'
                tgt_node = f'L{layer_idx+1}_Ch{col}'
                G.add_edge(src_node, tgt_node, weight=val)
    
    return G, pos

def visualize_connection_graph(G, pos, start_layer, weight_scale=5.0, min_width=0.3,
                               display_image=False, img_dir=None, img_size=300,
                               layer_names=LAYERS,
                               figsize=(15, 10), option='original_image',
                               show_channel_number=True):
    """
    Visualize the connection graph with visible arrows
    """
    plt.figure(figsize=figsize)
    
    # Get all edges and weights
    edges = G.edges(data=True)
    edge_list = [(u, v) for (u, v, d) in edges]
    
    # Get connected nodes
    connected_nodes = set()
    for u, v in edge_list:
        connected_nodes.add(u)
        connected_nodes.add(v)
    
    # Scale weights for edge widths
    weights = [G[u][v]['weight'] for (u, v) in edge_list]
    max_weight = max(weights) if weights else 1.0
    scaled_weights = [max(((w/max_weight) ** 1.5) * weight_scale, min_width) for w in weights]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_size=300,
                          node_color='lightgray',
                          alpha=0.5)
    
    # Draw edges with arrows
    if edge_list:
        nx.draw_networkx_edges(G, pos, 
                             edgelist=edge_list, 
                             edge_color='blue',
                             width=scaled_weights, 
                             alpha=0.7,
                             arrows=True,  # Enable arrows
                             arrowsize=15,  # Size of arrow heads
                             arrowstyle='->',  # Arrow style
                             min_source_margin=12,  # Space between arrow and source node
                             min_target_margin=20)  # Space between arrow and target node
    
    # Add labels only for connected nodes
    node_tmp = next(iter(connected_nodes))
    if isinstance(node_tmp, str):
        labels = {node: node.split('_')[1] for node in connected_nodes}
    else:
        labels = {node: node[1] for node in connected_nodes}

    nx.draw_networkx_labels(G, pos, labels, 
                          font_size=12,
                          font_color='black',
                          alpha=0.7)

    start_layer = start_layer.replace('_block', '.')
    start_layer_idx = layer_names.index(start_layer)
    
    # Retrieve nodes from edges
    # Add images to nodes
    if display_image:
        ax = plt.gca()
        nodes = set()
        for edge in edge_list:
            nodes.add(edge[0])
            nodes.add(edge[1])
            
        for node in nodes:
            if isinstance(node, str):
                node_tmp = node.replace('L', '').replace('Ch', '')
                layer_idx, cid = node_tmp.split('_')
                layer_idx = start_layer_idx + int(layer_idx)
                cid = int(cid)
                layer_name = layer_names[layer_idx]
                # print(node, layer_idx, cid, layer_name)
            else:
                layer_name = f"layer{G.nodes[node]['layer']}"
                cid = G.nodes[node]['cid']
                # print(node, layer_name, cid)
    
            # img = load_image(os.path.join(img_dir, layer_name, option, f'{layer_name}_{cid:04d}_cropped_image.png'), img_size)
            img = load_image(os.path.join(img_dir, layer_name, option, f'{cid:04d}.png'), img_size)
            img = OffsetImage(img, zoom=0.2)  # Adjust zoom as needed
            ab = AnnotationBbox(img, pos[node], frameon=False)
            ax.add_artist(ab)

            # Add text to nodes (channel number)
            if show_channel_number:
                text = ax.text(pos[node][0], pos[node][1], G.nodes[node]['cid'], ha='center', va='center', fontsize=8, color='black')
                bbox_props = dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='white', alpha=0.7)
                text.set_bbox(bbox_props)


    # ch_idx = pkl_path.split('_')[-1].split('.')[0]
    # plt.title(f"Channel Connections from {layer_idx} {ch_idx}", fontsize=18)
    plt.axis('on')
    plt.grid(True, linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    plt.show()


def longest_common_path_from_connections(*connections):
    def lcp(src_node, tgt_node, memo, *conns):
        if (src_node, tgt_node) in memo:
            return memo[(src_node, tgt_node)]

        if any(src_node not in conn for conn in conns):
            return []

        if src_node == tgt_node:
            return []    

        next_nodes_sets = [set((n[0], n[1]) for n in conn.get(src_node, [])) for conn in conns]
        next_nodes = sorted(set.intersection(*next_nodes_sets))

        best_path = []
        for next_node in next_nodes:
            current_path = [(src_node, next_node)] + lcp(next_node, tgt_node, memo, *conns)
            if len(current_path) > len(best_path):
                best_path = current_path

        memo[(src_node, tgt_node)] = best_path
        return best_path

    # 그래프 생성 및 위상 정렬
    graphs = [create_graph_from_connections(conn, img_dir=None, img_size=300, option='cropped_image') for conn in connections]
    topos = [list(nx.topological_sort(G)) for G in graphs]
    print('topos', topos)

    # 공통 노드 찾기
    common_nodes = sorted(set.intersection(*map(set, topos)))

    # 메모이제이션
    memo = {}

    # 최장 경로 초기화
    longest_path = []

    # 공통 노드 쌍을 순회하며 최장 공통 경로를 찾는다.
    for src_node in common_nodes:
        for tgt_node in common_nodes:
            if src_node == tgt_node:
                continue
            
            current_path = lcp(src_node, tgt_node, memo, *connections)
            if len(current_path) > len(longest_path):
                longest_path = current_path

    return longest_path


def longest_common_path_from_graph(G1, G2):
    def lcp(src_node, tgt_node, memo, G1, G2):
        # 종료 조건: 이미 탐색한 경로이면 종료
        if (src_node, tgt_node) in memo:
            return memo[(src_node, tgt_node)]

        # 종료 조건: 소스 노드가 리프 노드이면 탐색 종료
        if (src_node not in G1) or (src_node not in G2):
            return []

        # 종료 조건: 소스 노드와 타겟 노드가 같으면 탐색 종료
        if src_node == tgt_node:
            return []    

        nodes1 = [(n[0], n[1]) for n in G1[src_node]]
        nodes2 = [(n[0], n[1]) for n in G2[src_node]]

        next_nodes = set(nodes1) & set(nodes2)
        next_nodes = sorted(next_nodes)

        best_path = []
        for next_node in next_nodes:
            current_path = [(src_node, next_node)] + lcp(next_node, tgt_node, memo, G1, G2)
            if len(current_path) > len(best_path):
                best_path = current_path

        memo[(src_node, tgt_node)] = best_path
        return best_path

    # 위상 정렬
    topo1 = list(nx.topological_sort(G1))
    topo2 = list(nx.topological_sort(G2))
    # print('topo1', topo1)
    # print('topo2', topo2)

    # 공통 노드 찾기
    common_nodes = set(topo1) & set(topo2)
    common_nodes = sorted(common_nodes)
    # print('common_nodes', common_nodes)

    # 메모이제이션: 중복되는 경로 탐색 방지. key = (src_node, trg_node)
    memo = {}

    # 최장 경로 초기화
    longest_path = []

    # print('='*50)
    # 공통 노드 쌍을 순회하며 최장 공통 경로를 찾는다.
    for src_node in common_nodes:
        for tgt_node in common_nodes:
            # 출발 노드와 도착 노드가 같으면 탐색하지 않는다.
            if src_node == tgt_node:
                continue
            
            # 최장 공통 경로 탐색: 소스 노드에서 타겟 노드까지의 최장 공통 경로를 재귀적으로 탐색
            current_path = lcp(src_node, tgt_node, memo, G1, G2)
            if len(current_path) > len(longest_path):
                # print(f"Update prev: {longest_path} -> {current_path}")
                longest_path = current_path

    return longest_path


if __name__ == '__main__':
    img_dir_name = '/project/PURE/results/global_features/imagenet/resnet50_torchvision'
    
    pkl_dir_name = '/project/PURE/results_circuit/resnet50/imagenet'
    pkl_paths = glob.glob(os.path.join(pkl_dir_name, '**', 'connection_matrices_from_*.pkl'), recursive=True)
    pkl_paths = sorted(pkl_paths)
    for pkl_path in tqdm(pkl_paths):
        # print(pkl_path)
        visualize_from_pkl(pkl_path, img_dir_name, img_size=300, is_save=True)