import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import pandas as pd
import os
import cv2
pd.options.mode.chained_assignment = None
plt.rcParams.update({'font.size': 16})
# Extract and organise data from .csv files
current_dir = os.path.dirname(__file__)
# Node region allocation file
node_data = pd.read_csv(os.path.join(current_dir, 'data', 'Categorized_Brain_Nodes.csv'))
# Edge weight file
weight_data = pd.read_csv(os.path.join(current_dir, 'data', 'weights.csv'), header = None)

# Categorise nodes into super_regions
node_data = node_data.sort_values(by = ['Super_Region'])
super_regions = list(dict.fromkeys(node_data['Super_Region']))
categorized_regions = [0,0,0,0,0,0,0]
for idx, i in enumerate(super_regions):
    categorized_regions[idx] = node_data[node_data['Super_Region'] == i]
print(categorized_regions)
# Calculate cumualitve sum of number of nodes in each region
# For later use in indexing
region_counts = []
for i in categorized_regions:
    region_counts.append(len(i.index))
region_counts = np.cumsum(region_counts)
# Number of nodes
num_nodes = len(node_data.index)

# Define model parameters
model_choice = input("Select model: fk (Fisher-Kolmogorov), (h) Heterodimer, (s) Smoluchowski")
beta_no_damage = 0
gamma_no_damage = 0
beta_severe = 1/4
gamma_severe = 1 / 8 
beta_unrealistic = 4 
gamma_unrealistic = 2
if model_choice.lower() == "fk":
    alpha = 3/4 # 3/4 per year
elif model_choice.lower() == "h":
    k0 = 1
    k1 = 0.5
    k2 = 0.5
    k12 = 0.5
    alpha = k12*(k0/k1)-k2
  
rho = 1/100  # mm/yr, Laplacian rate constant

# Initialize c_k (toxic protein concentration), q_k (node damage), and w_kj (edge weight)
# For Heterodimer model, no c_k, p_k = healthy protein concentration pp_k = unhealthy protein concentration
p_k = np.full(num_nodes, 2)
pp_k = np.zeros(num_nodes)
c_k = np.zeros(num_nodes)
q_k = np.zeros(num_nodes)
w_kj = weight_data.to_numpy()
#Get indexes for entorhinal nodes
seed_nodes = node_data[node_data['Region'] == 'entorhinal']
# Set the initial toxic protein concentration for the entorhinal nodes
# c_k = 0.025 for fk model and pp_k = 0.5 for h model
if model_choice.lower() == "fk":
    seed_protein_concentration = 0.025
    for i in seed_nodes.index:
        c_k[i] = seed_protein_concentration
elif model_choice.lower() == "h":
    seed_protein_concentration = 0.5
    for i in seed_nodes.index:
        pp_k[i] = seed_protein_concentration

# Define the differential equation model
# Equations for fk model
if model_choice.lower() == 'fk':
    def dcdt(c_k, q_k, w_kj, alpha):
            D = np.diag(w_kj.sum(axis=1))
            L = rho * (D - w_kj)
            return -L.dot(c_k) + alpha * c_k * (1 - c_k)
        
    def dqdt(c_k, q_k, beta):
        return beta * c_k * (1 - q_k)

    def dwdt(w_kj, q_k, gamma):
        q_sum = q_k[:, None] + q_k  # Calculate the sum of damage
        return -gamma * w_kj * q_sum
# Equations for Heterodimer model
elif model_choice.lower() == 'h':
    def dpdt(p_k, pp_k, w_kj):
        D = np.diag(w_kj.sum(axis=1))
        L = rho * (D - w_kj)
        return -L.dot(p_k) + k0 - k1*p_k - k12*p_k*pp_k #Healthy protein concentration
    
    def dppdt(p_k, pp_k, w_kj):
        D = np.diag(w_kj.sum(axis=1))
        L = rho * (D - w_kj) 
        return -L.dot(pp_k) - k2*pp_k + k12*p_k*pp_k # Unhealthy protein concentration
    
    def dqdt(pp_k, q_k, beta):
        return beta * pp_k * (1 - q_k) # Node damage

    def dwdt(w_kj, q_k, gamma):
        q_sum = q_k[:, None] + q_k  # Calculate the sum of damage
        return -gamma * w_kj * q_sum # New weights
# Integrate the model equations
if model_choice.lower() == 'fk':
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
    # Set initial conditions
    y0 = np.concatenate((c_k, q_k, w_kj.flatten()))
elif model_choice.lower() == 'h':
# Integrate the model equations
    def model(y, t, beta, gamma):
        num_variables = 4 * num_nodes + num_nodes**2
        p_k = y[:num_nodes]
        pp_k = y[num_nodes:num_nodes*2]
        q_k = y[num_nodes*2:num_nodes*3]
        w_kj = y[3*num_nodes:num_variables].reshape((num_nodes, num_nodes))
        
        # Calculate dp/dt, dpp/dt, dq/dt, dw/dt
        dp_dt = dpdt(p_k, pp_k, w_kj)
        dpp_dt = dppdt(p_k, pp_k, w_kj)
        dq_dt = dqdt(pp_k, q_k, beta)
        dw_dt = dwdt(w_kj, q_k, gamma).flatten()
        
        # Merge them back into a flat array
        dydt = np.concatenate((dp_dt, dpp_dt, dq_dt, dw_dt))
        return dydt
    # Set initial conditions
    y0 = np.concatenate((p_k, pp_k, q_k, w_kj.flatten()))
# Set the time points
t = np.linspace(0, 30, 1000) # From 0 to 30 years, a total of 1000 time points

# No damage (β=γ=0)
sol_no_damage = odeint(model, y0, t, args=(beta_no_damage, gamma_no_damage))
if model_choice.lower() == 'fk':
    c_k_no_damage = sol_no_damage[:, :num_nodes]

    C_T_no_damage = c_k_no_damage.mean(axis=1)
elif model_choice.lower() == 'h':
    pp_k_no_damage = sol_no_damage[:, num_nodes:num_nodes*2]

    PP_T_no_damage = pp_k_no_damage.mean(axis=1)

# Severe damage (β=1/4, γ=1/8)

sol_severe = odeint(model, y0, t, args=(beta_severe, gamma_severe))
if model_choice.lower() == 'fk':
    c_k_severe = sol_severe[:, :num_nodes]
    q_k_severe = sol_severe[:, num_nodes:2*num_nodes]
    w_kj_severe = sol_severe[:, 2*num_nodes:].reshape((len(t), num_nodes, num_nodes))

    C_T_severe = c_k_severe.mean(axis=1)
    Q_severe = q_k_severe.mean(axis=1)
    W_severe = np.array([np.linalg.norm(w) / np.linalg.norm(w_kj) for w in w_kj_severe])

elif model_choice.lower() == 'h':
    p_k_severe = sol_severe[:, :num_nodes]
    pp_k_severe = sol_severe[:, num_nodes:num_nodes*2]
    q_k_severe = sol_severe[:, num_nodes*2:num_nodes*3]
    w_kj_severe = sol_severe[:, num_nodes*3:].reshape((len(t), num_nodes, num_nodes))

    P_T_severe = p_k_severe.mean(axis=1)
    PP_T_severe = pp_k_severe.mean(axis=1)
    Q_severe = q_k_severe.mean(axis=1)
    W_severe = np.array([np.linalg.norm(w) / np.linalg.norm(w_kj) for w in w_kj_severe])

# Unrealistic damage (β=4, γ=2)
sol_unrealistic = odeint(model, y0, t, args=(beta_unrealistic, gamma_unrealistic))
if model_choice.lower() == 'fk':
    c_k_unrealistic = sol_unrealistic[:, :num_nodes]
    q_k_unrealistic = sol_unrealistic[:, num_nodes:2*num_nodes]
    w_kj_unrealistic = sol_unrealistic[:, 2*num_nodes:].reshape((len(t), num_nodes, num_nodes))

    C_T_unrealistic = c_k_unrealistic.mean(axis=1)
    Q_unrealistic = q_k_unrealistic.mean(axis=1)
    W_unrealistic = np.array([np.linalg.norm(w) / np.linalg.norm(w_kj) for w in w_kj_unrealistic])
elif model_choice.lower() == 'h':
    p_k_unrealistic = sol_unrealistic[:, :num_nodes]
    pp_k_unrealistic = sol_unrealistic[:, num_nodes:num_nodes*2]
    q_k_unrealistic = sol_unrealistic[:, num_nodes*2:num_nodes*3]
    w_kj_unrealistic = sol_unrealistic[:, num_nodes*3:].reshape((len(t), num_nodes, num_nodes))

    P_T_unrealistic = p_k_unrealistic.mean(axis=1)
    PP_T_unrealistic = pp_k_unrealistic.mean(axis=1)
    Q_unrealistic = q_k_unrealistic.mean(axis=1)
    W_unrealistic = np.array([np.linalg.norm(w) / np.linalg.norm(w_kj) for w in w_kj_unrealistic])


# Regional data
# Change Model data to observe regional damage in different cases e.g. Severe and Unrealistic
# Dividing up q_k into respective regions
if model_choice.lower() == 'fk':
    model_data = sol_severe[:,num_nodes:num_nodes*2]
    basal_ganglia_q_k = model_data[:, node_data[node_data['Super_Region'] == 'basal_ganglia'].index]
    brain_stem_q_k = model_data[:, node_data[node_data['Super_Region'] == 'brain_stem'].index]
    frontal_q_k = model_data[:, node_data[node_data['Super_Region'] == 'frontal'].index]
    limbic_q_k = model_data[:, node_data[node_data['Super_Region'] == 'limbic'].index]
    occipital_q_k = model_data[:, node_data[node_data['Super_Region'] == 'occipital'].index]
    parietal_q_k = model_data[:, node_data[node_data['Super_Region'] == 'parietal'].index]
    temporal_q_k = model_data[:, node_data[node_data['Super_Region'] == 'temporal'].index]
elif model_choice.lower() == 'h':
    model_data = sol_severe[:,num_nodes*2:num_nodes*3]
    basal_ganglia_q_k = model_data[:, node_data[node_data['Super_Region'] == 'basal_ganglia'].index]
    brain_stem_q_k = model_data[:, node_data[node_data['Super_Region'] == 'brain_stem'].index]
    frontal_q_k = model_data[:, node_data[node_data['Super_Region'] == 'frontal'].index]
    limbic_q_k = model_data[:, node_data[node_data['Super_Region'] == 'limbic'].index]
    occipital_q_k = model_data[:, node_data[node_data['Super_Region'] == 'occipital'].index]
    parietal_q_k = model_data[:, node_data[node_data['Super_Region'] == 'parietal'].index]
    temporal_q_k = model_data[:, node_data[node_data['Super_Region'] == 'temporal'].index]
basal_ganglia_Q = basal_ganglia_q_k.mean(axis=1)
basal_ganglia_Q = basal_ganglia_q_k.mean(axis=1)
brain_stem_Q = brain_stem_q_k.mean(axis=1)
frontal_Q = frontal_q_k.mean(axis=1)
limbic_Q = limbic_q_k.mean(axis=1)
occipital_Q = occipital_q_k.mean(axis=1)
parietal_Q = parietal_q_k.mean(axis=1)
temporal_Q = temporal_q_k.mean(axis=1)

# Plot the results
'''
#Fig 2
plt.figure(figsize=(12, 6))
if model_choice.lower() == 'fk':
    plt.plot(t, C_T_no_damage, 'black', linestyle="--",linewidth=3, label='Average concentration in the absence of damage')
    plt.plot(t, C_T_severe, 'b-', label='C: Average concentration in the case of severe damage')
    plt.plot(t, C_T_unrealistic, 'g--',linewidth=3, label='C: Average concentration in the case of unrealistic damage')
elif model_choice.lower() == 'h':
    plt.plot(t, PP_T_no_damage, 'black', linestyle="--",linewidth=3, label='Average concentration in the absence of damage')
    plt.plot(t, PP_T_severe, 'b-', label='C: Average concentration in the case of severe damage')
    plt.plot(t, PP_T_unrealistic, 'g--',linewidth=3, label='C: Average concentration in the case of unrealistic damage')    
plt.plot(t, Q_severe, color="purple", label='Q: Node damage in the case of severe damage')
plt.plot(t, Q_unrealistic, color="purple", linestyle="--", label='Q: Node damage in the case of unrealistic damage')
plt.plot(t, W_severe, 'r-', label='W: Edge weight in the case of severe damage')
plt.plot(t, W_unrealistic, 'r--', label='W: Edge weight in the case of unrealistic damage')
plt.title('Evolution of averaged toxic concentration, node damage, and edge weight')
plt.xlabel('Time (yr)')
plt.ylabel('Concentration/Damage/Weight')
plt.legend()
plt.tight_layout()
'''
#Fig 3
plt.figure(figsize=(10, 6))
plt.title("Regional Damage accumulation")
plt.plot(t, frontal_Q, color="red", label='frontal')
plt.plot(t, parietal_Q, color="orange", label='parietal')
plt.plot(t, occipital_Q, color="greenyellow", label='occipital')
plt.plot(t, temporal_Q, color="deepskyblue", label='temporal')
plt.plot(t, limbic_Q, color="dodgerblue", label='limbic')
plt.plot(t, basal_ganglia_Q, color="blue", label='basal ganglia')
plt.plot(t, brain_stem_Q, color="black", label='brain stem')
plt.xlabel('Time (yr)')
plt.ylabel('Damage')
plt.legend()
plt.tight_layout()
#Fig 4

seed_node_testing = []
stepsize = 1
for i in range(0, 10, stepsize):
    seed_protein_concentration = i*0.1+0.1
    for j in seed_nodes.index:
        c_k[j] = seed_protein_concentration
    y0 = np.concatenate((c_k, q_k, w_kj.flatten()))
    sol_severe = odeint(model, y0, t, args=(beta_severe, gamma_severe))
    q_k_severe = sol_severe[:, num_nodes:2*num_nodes]
    regional_results = []
    regional_results.append(q_k_severe[:, node_data[node_data['Super_Region'] == 'basal_ganglia'].index])
    regional_results.append(q_k_severe[:, node_data[node_data['Super_Region'] == 'brain_stem'].index])
    regional_results.append(q_k_severe[:, node_data[node_data['Super_Region'] == 'frontal'].index])
    regional_results.append(q_k_severe[:, node_data[node_data['Super_Region'] == 'limbic'].index])
    regional_results.append(q_k_severe[:, node_data[node_data['Super_Region'] == 'occipital'].index])
    regional_results.append(q_k_severe[:, node_data[node_data['Super_Region'] == 'parietal'].index])
    regional_results.append(q_k_severe[:, node_data[node_data['Super_Region'] == 'temporal'].index])
    for idx, k in enumerate(regional_results):
        regional_results[idx] = k.mean(axis=1)
    seed_node_testing.append(regional_results)
seed_node_testing = np.transpose(seed_node_testing, (1,0,2))
plt.figure(figsize=(10, 6))
plt.title("Average Node Damage Over Time Based On Seed Protein Concentration")
colours = ["blue", "black", "red", "dodgerblue", "greenyellow", "orange", "deepskyblue"]
print(np.shape(seed_node_testing))
for idx, i in enumerate(seed_node_testing):
    for idx2, j in enumerate(i):
        plt.plot(t, j, color=colours[idx], alpha = ((idx2+1)*0.1*stepsize))
plt.xlabel('Time (yr)')
plt.ylabel('Damage')
frontalh = Line2D([0], [0], label='frontal', color='red')
parietalh = Line2D([0], [0], label='parietal', color='orange')
occipitalh = Line2D([0], [0], label='occipital', color='greenyellow')
temporalh = Line2D([0], [0], label='temporal', color='deepskyblue')
limbich = Line2D([0], [0], label='limbic', color='dodgerblue')
basal_gangliah = Line2D([0], [0], label='basal ganglia', color='blue')
brain_stemh = Line2D([0], [0], label='brain stem', color='black')
handles = [frontalh, parietalh, occipitalh, temporalh, limbich, basal_gangliah, brain_stemh]
plt.legend(handles = handles)
plt.tight_layout()

seed_node_testing = []
stepsize = 1
for i in range(0, 10, stepsize):
    seed_protein_concentration = i*0.1+0.1
    for j in seed_nodes.index:
        c_k[j] = seed_protein_concentration
    y0 = np.concatenate((c_k, q_k, w_kj.flatten()))
    sol_severe = odeint(model, y0, t, args=(beta_severe, gamma_severe))
    q_k_severe = sol_severe[:, num_nodes:2*num_nodes]
    seed_node_testing.append(q_k_severe.mean(axis=1))
plt.figure(figsize=(10, 6))
plt.title("Global Node Damage Based On Seed Protein Concentration")
for idx, i in enumerate(seed_node_testing):
    plt.plot(t, i, color=mpl.cm.winter(idx*0.1*stepsize+0.1))
plt.xlabel('Time (yr)')
plt.ylabel('Damage')
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.winter, orientation='vertical')
plt.gcf().add_axes(ax_cb)
plt.tight_layout()
plt.show()


