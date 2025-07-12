import pennylane as qml
from pennylane import numpy as np  # Important: use PennyLane's NumPy wrapper for Autograd
import networkx as nx
import matplotlib.pyplot as plt
import random

#randomly generated graph and number of wires
num_wires = 3  # Randomly choose number of wires (3 to 10)
wires = range(num_wires)
edges = []
for i in range(num_wires):
    # Ensure each node connects to at least one other node
    possible = [j for j in range(num_wires) if j != i]
    j = random.choice(possible)
    edge = tuple(sorted((i, j)))
    edges.append(edge)

# Add more random edges, avoiding duplicates
for i in range(num_wires):
    for j in range(i + 1, num_wires):
        if random.random() < 0.3:  # 30% probability for extra edge
            edge = (i, j)
            edges.append(edge)
print(f"Generated graph with {num_wires} wires and edges: {edges}")
graph = nx.Graph()
graph.add_nodes_from(wires)  # Add all wires as graph nodes
graph.add_edges_from(edges)

# Step 2: Create QAOA cost and mixer Hamiltonians from the graph
cost_h, mixer_h = qml.qaoa.maxcut(graph)
print("Cost Hamiltonian:", cost_h)
print("Mixer Hamiltonian:", mixer_h)

# Step 3: Define a single QAOA layer (cost + mixer)
def qaoa_layer(gamma, alpha):
    qml.qaoa.cost_layer(gamma, cost_h)   # Apply cost Hamiltonian as a unitary
    qml.qaoa.mixer_layer(alpha, mixer_h) # Apply mixer Hamiltonian as a unitary

# Step 4: Construct the full QAOA circuit with p=2 layers
def circuit(params):
    for w in wires:
        qml.Hadamard(wires=w)  # Start in equal superposition state |+>

    # Apply each QAOA layer with its (gamma, alpha) parameters
    for gamma, alpha in params:
        qaoa_layer(gamma, alpha)

# Step 5: Define quantum device and QNode to evaluate cost expectation
dev = qml.device("default.qubit", wires=len(wires))

@qml.qnode(dev)
def cost_function(params):
    circuit(params)
    return qml.expval(cost_h) # Expectation value of cost Hamiltonian

# Step 6: Optimize the parameters using a classical optimizer
opt = qml.AdagradOptimizer(stepsize=0.5)

p = 2  # Number of QAOA layers
init_params = np.array([[0.5, 0.5]] * p, requires_grad=True)  # Initial parameters
params = init_params

steps = 20
tolerance = 0.005  # Set your threshold here
prev_cost = cost_function(params)

for i in range(steps):
    params = opt.step(cost_function, params)
    cost_val = cost_function(params)
    if (i + 1) % 5 == 0:
        print(f"Iteration {i + 1}: Cost = {cost_val:.4f}")
    if abs(cost_val - prev_cost) < tolerance:
        print(f"Converged at iteration {i + 1}: Cost = {cost_val:.4f}")
        break
    prev_cost = cost_val
else:
    print(f"Final Cost after {steps} iterations: {cost_val:.4f}")

print("\nOptimized Parameters:")
print(params)

# Step 7: Sample the final circuit to retrieve solutions
@qml.qnode(dev)
def sample_output(params):
    circuit(params)
    return [qml.sample(qml.PauliZ(w)) for w in wires]  # Measure in Z basis

n_samples = 1000  # More samples for better statistics
bitstrings_raw = sample_output(params, shots=n_samples)

# Transpose the sample matrix
bitstrings_transposed = np.transpose(bitstrings_raw)

# Convert PauliZ outcomes (+1/-1) to bitstrings (0/1) and count frequencies
counts = {}
for sample in bitstrings_transposed:
    bitstring = "".join([str(int((1 - s) / 2)) for s in sample])  # Convert to '0'/'1'
    counts[bitstring] = counts.get(bitstring, 0) + 1

print(counts)

# Identify the most frequent bitstring (probable solution)
most_freq_bitstring = max(counts, key=counts.get)
print(f"\nMost frequently sampled bitstring: {most_freq_bitstring}")
print("Sample counts:", counts)

# Step 8: Visualize the graph with solution coloring
coloring = [int(bit) for bit in most_freq_bitstring]
node_colors = ['gold' if bit == 0 else 'lightblue' for bit in coloring]

G = nx.Graph()
G.add_nodes_from(wires)
G.add_edges_from(edges)
nx.draw(G, with_labels=True, node_color=node_colors, node_size=800, font_size=16)
plt.title("Graph Partitioned by QAOA (MaxCut)")

# plot the most frequent bitstring as a bar chart
plt.figure(figsize=(10, 5))
plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.xlabel('Bitstrings')
plt.ylabel('Counts')
plt.title('Sampled Bitstrings from QAOA Circuit')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()