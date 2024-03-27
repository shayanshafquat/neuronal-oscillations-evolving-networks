import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

# Load the data from the provided CSV file
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'data', 'graph_data.csv')
data = pd.read_csv(file_path, delimiter=';')

# Define model parameters
alpha = 0.75
beta_severe = 1 / 4
gamma_severe = 1 / 8
rho = 1 / 100

# Extract the necessary information
nodes = np.union1d(data['id node1'].values, data['id node2'].values)
num_nodes = len(nodes)
edges = list(zip(data['id node1'], data['id node2']))

# Create a mapping from node ID to index
node_to_index = {node_id: index for index, node_id in enumerate(nodes)}

# Initialize c_k (toxic protein concentration), q_k (node damage), and w_kj (edge weight)
c_k = np.zeros(num_nodes)
q_k = np.zeros(num_nodes)
w_kj = np.zeros((num_nodes, num_nodes))

# Set the initial toxic protein concentration for the entorhinal nodes

# Assume that the node indexes corresponding to the medial frontal area are 26 and 68
entorhinal_nodes = [26, 68]
seed_protein_concentration = 0.025
for entorhinal_node in entorhinal_nodes:
    if entorhinal_node in node_to_index: # Check if the node exists
        c_k[node_to_index[entorhinal_node]] = seed_protein_concentration

for _, row in data.iterrows():
    idx1 = node_to_index[row['id node1']]
    idx2 = node_to_index[row['id node2']]
    weight = row['edge weight(med nof)']
    w_kj[idx1, idx2] = weight
    w_kj[idx2, idx1] = weight

# Check initialization status
(c_k, q_k, w_kj[:5, :5])  # Display only a small part of w_kj to verify initialization status

# Define the differential equation model
def dcdt(c_k, q_k, w_kj, alpha):
    D = np.diag(w_kj.sum(axis=1))  
    L = rho * (D - w_kj) 
    return -L.dot(c_k) + alpha * c_k * (1 - c_k)

def dqdt(c_k, q_k, beta):
    return beta * c_k * (1 - q_k)

def dwdt(w_kj, q_k, gamma):
    q_sum = q_k[:, None] + q_k  # Calculate the sum of damage
    return -gamma * w_kj * q_sum

# Integrate the model equations
def model(y, t, beta, gamma):
    num_variables = 2 * num_nodes + num_nodes**2
    c_k = y[:num_nodes]
    q_k = y[num_nodes:2*num_nodes]
    w_kj = y[2*num_nodes:num_variables].reshape((num_nodes, num_nodes))
    
    # Calculate dc/dt, dq/dt, dw/dt
    dc_dt = dcdt(c_k, q_k, w_kj, alpha)
    dq_dt = dqdt(c_k, q_k, beta)
    dw_dt = dwdt(w_kj, q_k, gamma).flatten()
    
    # Merge them back into a flat array
    dydt = np.concatenate((dc_dt, dq_dt, dw_dt))
    return dydt

# Set the time points
t = np.linspace(0, 30, 1000) # From 0 to 30 years, a total of 1000 time points

# Set initial conditions
y0 = np.concatenate((c_k, q_k, w_kj.flatten()))

# Solve the model under severe damage conditions
sol_severe = odeint(model, y0, t, args=(beta_severe, gamma_severe))
q_k_severe = sol_severe[:, num_nodes:2 * num_nodes]

# Update regex to more accurately reflect classification
# Note: Consider specific region names such as parahippocampal in both temporal and limbic
node_regions = {}
for _, row in data.iterrows():
    node1_name = row['name node1'].lower()  
    node2_name = row['name node2'].lower()

    # Define regular expression patterns for each brain region
    patterns = {
        'frontal': 'frontal|orbitofrontal|precentral|rostralanteriorcingulate|caudalanteriorcingulate',
        'parietal': 'parietal|postcentral|supramarginal|precuneus',
        'occipital': 'occipital|lingual|pericalcarine',
        'temporal': 'temporal|superiortemporal|middletemporal|inferiortemporal|fusiform|parahippocampal|entorhinal|temporalpole',
        'limbic': 'cingulate|isthmuscingulate|parahippocampal|entorhinal',
        'basalganglia': 'caudate|putamen|pallidum|accumbens',
        'brainstem': 'brain-stem'
    }

    #Initialize the area to an empty list
    node1_regions = []
    node2_regions = []

    # Check each area to determine which area the node belongs to
    for region, pattern in patterns.items():
        if re.search(pattern, node1_name):
            node1_regions.append(region)

    for region, pattern in patterns.items():
        if re.search(pattern, node2_name):
            node2_regions.append(region)

    node_regions[row['id node1']] = node1_regions
    node_regions[row['id node2']] = node2_regions

#Group by area and calculate the average damage value of nodes in each area
region_groups = {}
for region in patterns.keys():
    region_nodes = [node for node, node_regions in node_regions.items() if region in node_regions]
    if region_nodes:
        region_indices = [node_to_index[node] for node in region_nodes if node in node_to_index] 
        region_groups[region] = q_k_severe[:, region_indices].mean(axis=1) if region_indices else np.zeros_like(t)
    else:
        region_groups[region] = np.zeros_like(t)

# Draw the average node damage curve for each region
plt.figure(figsize=(12, 6))
for region, damage in region_groups.items():
    color = 'red' if region == 'frontal' else \
            'orange' if region == 'parietal' else \
            'green' if region == 'occipital' else \
            'cyan' if region == 'temporal' else \
            'blue' if region == 'limbic' else \
            'navy' if region == 'basalganglia' else \
            'black' if region == 'brainstem' else \
            'gray'  #Default color
        
    plt.plot(t, damage, label=region, color=color)
plt.title('Evolution of damage in different brain regions')
plt.xlabel('Time (yr)')
plt.ylabel('Average node damage')
plt.legend()
plt.tight_layout()
plt.show()
