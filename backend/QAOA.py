from flask import Flask, jsonify, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml
from pennylane import numpy as np
import random
import io
import base64
plt.style.use('dark_background')


app = Flask(__name__)

@app.route('/api/qaoa', methods=['POST'])
def run_qaoa():
    # Random graph setup
    num_wires = random.randint(3, 5)
    wires = range(num_wires)
    edges = []
    for i in range(num_wires):
        possible = [j for j in range(num_wires) if j != i]
        j = random.choice(possible)
        edges.append(tuple(sorted((i, j))))
    for i in range(num_wires):
        for j in range(i + 1, num_wires):
            if random.random() < 0.3:
                edges.append((i, j))

    graph = nx.Graph()
    graph.add_nodes_from(wires)
    graph.add_edges_from(edges)

    # Create QAOA cost and mixer Hamiltonians from the graph
    cost_h, mixer_h = qml.qaoa.maxcut(graph)

    # Define a single QAOA layer (cost + mixer)
    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(alpha, mixer_h)

    # Construct the full QAOA circuit with p layers
    def circuit(params):
        for w in wires:
            qml.Hadamard(wires=w)
        for gamma, alpha in params:
            qaoa_layer(gamma, alpha)

    # Define quantum device and QNode to evaluate cost expectation
    dev = qml.device("default.qubit", wires=len(wires))

    @qml.qnode(dev)
    def cost_function(params):
        circuit(params)
        return qml.expval(cost_h)

    # Optimize the parameters using a classical optimizer
    opt = qml.AdagradOptimizer(stepsize=0.5)
    p = 8 # Number of QAOA layers
    params = np.array([[0.5, 0.5]] * p, requires_grad=True)
    steps = 50
    tolerance = 0.005
    prev_cost = cost_function(params)
    iteration_logs = []

    for i in range(steps):
        params = opt.step(cost_function, params)
        cost_val = cost_function(params)
        if (i + 1) % 5 == 0:
            log_line = f"{i + 1:>2}: Cost = {cost_val:.4f}"
            iteration_logs.append(log_line)
        if abs(cost_val - prev_cost) < tolerance:
            log_line = f"Converged at iteration {i + 1}: Cost = {cost_val:.4f}"
            iteration_logs.append(log_line)
            break
        prev_cost = cost_val

            

    optimized_params = params.tolist()

    # Sample the final circuit to retrieve solutions
    @qml.qnode(dev)
    def sample_output(params):
        circuit(params)
        return [qml.sample(qml.PauliZ(w)) for w in wires]

    bitstrings_raw = sample_output(params, shots=1000)
    bitstrings_transposed = np.transpose(bitstrings_raw)

    # Convert PauliZ outcomes (+1/-1) to bitstrings (0/1) and count frequencies
    counts = {}
    for sample in bitstrings_transposed:
        bitstring = "".join([str(int((1 - s) / 2)) for s in sample])
        counts[bitstring] = counts.get(bitstring, 0) + 1

    # Identify the most frequent bitstring (probable solution)
    most_freq_bitstring = max(counts, key=counts.get)


    # Visualize the graph with solution coloring
    coloring = [int(bit) for bit in most_freq_bitstring]
    node_colors = ['gold' if bit == 0 else 'lightblue' for bit in coloring]
    top_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20])

    NEON_BLUE = '#00f0ff'
    NEON_GREEN = '#00ffb3'
    NEON_ACCENT = '#11cdef'

    G = nx.Graph()
    G.add_nodes_from(wires)
    G.add_edges_from(edges)

    plt.style.use("dark_background")
    plt.figure(figsize=(5, 5))
    nx.draw(
        G,
        with_labels=True,
        node_color=node_colors,
        edge_color=NEON_ACCENT,
        node_size=800,
        font_size=16,
        font_color="black"
    )
    plt.title("Graph Partitioned by QAOA (MaxCut)", color=NEON_ACCENT)

    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    graph_img = base64.b64encode(buf1.read()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(max(len(top_counts) * 0.4, 4), 4))
    plt.bar(
        top_counts.keys(),
        top_counts.values(),
        color=NEON_GREEN,
        width=0.6,
        edgecolor=NEON_ACCENT
    )
    plt.xlabel('Bitstrings', color=NEON_BLUE)
    plt.ylabel('Counts', color=NEON_BLUE)
    plt.title('Most Frequently Sampled Bitstrings', color=NEON_ACCENT)
    plt.xticks(rotation=45, ha='right', color=NEON_BLUE)
    plt.yticks(color=NEON_BLUE)

    plt.tight_layout()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    hist_img = base64.b64encode(buf2.read()).decode('utf-8')
    plt.close()

    console_output = "Console Output:"
    return jsonify({
        "graph": graph_img,
        "hist": hist_img,
        "iterations": iteration_logs,
        "optimized_params": optimized_params,
        "most_freq_bitstring": most_freq_bitstring,
        "console_output": console_output
    })

if __name__ == '__main__':
    app.run(debug=True)
