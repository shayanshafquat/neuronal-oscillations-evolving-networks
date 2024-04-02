import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import os


class TransportModel:
    def __init__(self):
        # Define model parameters
        self.alpha = 0.75  # 3/4 per year
        self.rho = 1 / 100  # mm/yr, Laplacian rate constant

    def load_data(self):
        # Load the data from the provided CSV file
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data', 'graph_data.csv')
        data = pd.read_csv(file_path, delimiter=';')

        # Extract the necessary information
        nodes = np.union1d(data['id node1'].values, data['id node2'].values)
        self.num_nodes = len(nodes)
        edges = list(zip(data['id node1'], data['id node2']))

        # Create a mapping from node ID to index
        node_to_index = {node_id: index for index, node_id in enumerate(nodes)}

        # Initialize c_k (toxic protein concentration), q_k (node damage), and w_kj (edge weight)
        self.c_k = np.zeros(self.num_nodes)
        self.q_k = np.zeros(self.num_nodes)
        self.w_kj = np.zeros((self.num_nodes, self.num_nodes))
        
        # Set the initial toxic protein concentration for the entorhinal nodes
        entorhinal_nodes = [26, 68]
        seed_protein_concentration = 0.025
        for entorhinal_node in entorhinal_nodes:
            if entorhinal_node in self.node_to_index: # Check if the node exists
                self.c_k[node_to_index[entorhinal_node]] = seed_protein_concentration

        for _, row in data.iterrows():
            idx1 = node_to_index[row['id node1']]
            idx2 = node_to_index[row['id node2']]
            weight = row['edge weight(med nof)']
            self.w_kj[idx1, idx2] = weight
            self.w_kj[idx2, idx1] = weight

        # Set the time points
        self.t = np.linspace(0, 30, 1000) # From 0 to 30 years, a total of 1000 time points

        # Set initial conditions
        self.y0 = np.concatenate((self.c_k, self.q_k, self.w_kj.flatten()))

    # Define the differential equation model
    def dcdt(self):
        # Define the graph Laplacian matrix L
        D = np.diag(self.w_kj.sum(axis=1))  
        L = self.rho * (D - self.w_kj) 
        return -L.dot(self.c_k) + self.alpha * self.c_k * (1 - self.c_k)

    def dqdt(self):
        return self.beta * self.c_k * (1 - self.q_k)

    def dwdt(self):
        q_sum = self.q_k[:, None] + self.q_k  # Calculate the sum of damage
        return -self.gamma * self.w_kj * q_sum
    
    # Integrate the model equations
    def model(self, beta, gamma):
        num_variables = 2 * self.num_nodes + self.num_nodes**2
        c_k = y[:self.num_nodes]
        q_k = y[num_nodes:2*num_nodes]
        w_kj = y[2*num_nodes:num_variables].reshape((num_nodes, num_nodes))
        
        # Calculate dc/dt, dq/dt, dw/dt
        dc_dt = self.dcdt(c_k, q_k, w_kj, self.alpha)
        dq_dt = self.dqdt(c_k, q_k, beta)
        dw_dt = self.dwdt(w_kj, q_k, gamma).flatten()
        
        # Merge them back into a flat array
        dydt = np.concatenate((dc_dt, dq_dt, dw_dt))
        return dydt
    
    def no_damage(self):
        # No damage (β=γ=0)
        sol_no_damage = odeint(self.model, self.y0, self.t, args=(self.beta_no_damage, gamma_no_damage))
        c_k_no_damage = sol_no_damage[:, :num_nodes]
        C_T_no_damage = c_k_no_damage.mean(axis=1)

if __name__ == "__main__":
    tm = TransportModel()
    tm.load_data()

    beta=0
    gamma = 0
    # No damage (β=γ=0)
    sol_no_damage = odeint(tm.model, y0, t, args=(beta, gamma))
    c_k_no_damage = sol_no_damage[:, :num_nodes]
    C_T_no_damage = c_k_no_damage.mean(axis=1)

# Severe damage (β=1/4, γ=1/8)
sol_severe = odeint(model, y0, t, args=(beta_severe, gamma_severe))
c_k_severe = sol_severe[:, :num_nodes]
q_k_severe = sol_severe[:, num_nodes:2*num_nodes]
w_kj_severe = sol_severe[:, 2*num_nodes:].reshape((len(t), num_nodes, num_nodes))
C_T_severe = c_k_severe.mean(axis=1)
Q_severe = q_k_severe.mean(axis=1)
W_severe = np.array([np.linalg.norm(w) / np.linalg.norm(w_kj) for w in w_kj_severe])

# unrealistic damage (β=4, γ=2)
sol_unrealistic = odeint(model, y0, t, args=(beta_unrealistic, gamma_unrealistic))
c_k_unrealistic = sol_unrealistic[:, :num_nodes]
q_k_unrealistic = sol_unrealistic[:, num_nodes:2*num_nodes]
w_kj_unrealistic = sol_unrealistic[:, 2*num_nodes:].reshape((len(t), num_nodes, num_nodes))
C_T_unrealistic = c_k_unrealistic.mean(axis=1)
Q_unrealistic = q_k_unrealistic.mean(axis=1)
W_unrealistic = np.array([np.linalg.norm(w) / np.linalg.norm(w_kj) for w in w_kj_unrealistic])

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(t, C_T_no_damage, 'black', linestyle="--",linewidth=3, label='Average concentration in the absence of damage')
plt.plot(t, C_T_severe, 'b-', label='C: Average concentration in the case of severe damage')
plt.plot(t, C_T_unrealistic, 'g--',linewidth=3, label='C: Average concentration in the case of unrealistic damage')
plt.plot(t, Q_severe, color="purple", label='Q: Node damage in the case of severe damage')
plt.plot(t, Q_unrealistic, color="purple", linestyle="--", label='Q: Node damage in the case of unrealistic damage')
plt.plot(t, W_severe, 'r-', label='W: Edge weight in the case of severe damage')
plt.plot(t, W_unrealistic, 'r--', label='W: Edge weight in the case of unrealistic damage')
plt.title('Evolution of averaged toxic concentration, node damage, and edge weight')
plt.xlabel('Time (yr)')
plt.ylabel('Concentration/Damage/Weight')
plt.legend()
plt.tight_layout()
plt.show()