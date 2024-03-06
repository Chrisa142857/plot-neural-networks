import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Define the dimensions of the neural network layers
layer_dimensions = [9, 9, 9, 2]
node_size = 150
edge_alpha_min = 0.5
# Initialize NN weights and bias
nn = [
    np.random.randn(layer_dimensions[i+1], layer_dimensions[i])
    for i in range(len(layer_dimensions)-1)
]
nn_bias = [
    np.random.randn(layer_dimensions[i+1])
    for i in range(len(layer_dimensions)-1)
]
fig_height = 4
fig_width_ratio = 2.5
plt.figure(figsize=(fig_width_ratio*len(layer_dimensions), fig_height))
pred_node_pr = 0.1*fig_height/node_size
# Create a directed graph
G = nx.DiGraph()
node_biases = []
# Add nodes representing neurons
for layer, num_neurons in enumerate(layer_dimensions):
    for i in range(num_neurons):
        if layer == 3:
            G.add_node((layer, i), pos=(layer, i*layer_dimensions[-2]/layer_dimensions[-1]+layer_dimensions[-2]/(layer_dimensions[-1]*2)-(pred_node_pr*node_size/2+i*pred_node_pr*node_size)))
        else:
            G.add_node((layer, i), pos=(layer, i))
        if layer == 0:
            node_biases.append(0)
        else:
            node_biases.append(nn_bias[layer-1][i])
        
node_biases = np.array(node_biases)

edge_weights = []
# Add edges representing weights
for layer in range(len(layer_dimensions) - 1):
    for i in range(layer_dimensions[layer]):
        for j in range(layer_dimensions[layer + 1]):
            G.add_edge((layer, i), (layer + 1, j))
            edge_weights.append(nn[layer][j,i])
edge_weights = np.stack(edge_weights)
# Normalize edge weights for better visualization
min_weight = min(edge_weights.min(), node_biases.min())
max_weight = max(edge_weights.max(), node_biases.max())
norm = plt.Normalize(vmin=min_weight, vmax=max_weight)

# Plot the graph
pos = nx.get_node_attributes(G, 'pos')
edges = G.edges()
weights = [edge_weights[i].item() for i in range(len(edges))]
biases = [node_biases[i] for i in range(len(G.nodes()))]

# Color bar
colors = [plt.cm.Oranges_r(norm(max_weight-1*x/100)) if x>=0 else plt.cm.Blues_r(norm(-1*min_weight+x/100)) for x in range(int(min_weight*100), int(max_weight*100), 1)]
# Edge, node colors
edge_colors = [plt.cm.Oranges_r(norm(max_weight-1*w)) if w>=0 else plt.cm.Blues_r(norm(-1*min_weight+w)) for w in weights]
node_colors = [plt.cm.Oranges_r(norm(max_weight-1*b)) if b>=0 else plt.cm.Blues_r(norm(-1*min_weight+b)) for b in biases]
alpha = (np.abs(edge_weights)-np.abs(edge_weights).min())/(np.abs(edge_weights).max()-np.abs(edge_weights).min())
alpha[alpha<edge_alpha_min] = edge_alpha_min
# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, edgecolors='skyblue', linewidths=3, node_shape='s')

# Draw edges with weights as curves
nx.draw_networkx_edges(G, pos, edgelist=edges, style='--', edge_color=edge_colors, width=1.5, alpha=alpha, connectionstyle="arc,angleA=0, angleB=-180, armA=40, armB=40, rad=40")

mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
sm = plt.cm.ScalarMappable(cmap=mymap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='Weights, Bias')

plt.title('Neural Network Architecture')
plt.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
plt.axis('off')
plt.tight_layout()
plt.savefig('nnplot.svg')

