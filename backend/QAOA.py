
from flask import Flask, jsonify, request, send_file, make_response
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml
from pennylane import numpy as np
import random, io, base64, os
from uuid import uuid4
from PIL import Image
import tempfile
from pathlib import Path
plt.style.use('dark_background')

app = Flask(__name__)
CORS(app, resources={
  r"/api/*": {"origins": [
    "http://localhost:3000",
    "https://qaoa-maxcut.onrender.com",
    "https://qaoa.onrender.com"
  ]}
})


# Cross-platform temp dir for storing images
TMP_DIR = Path(tempfile.gettempdir()) / "qaoa_images"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.route('/api/qaoa', methods=['POST'])
def run_qaoa():
    try:
        # Graph setup
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

        cost_h, mixer_h = qml.qaoa.maxcut(graph)

        # QAOA layer
        def qaoa_layer(gamma, alpha):
            qml.qaoa.cost_layer(gamma, cost_h)
            qml.qaoa.mixer_layer(alpha, mixer_h)

        # Circuit definition
        def circuit(params):
            for w in wires:
                qml.Hadamard(wires=w)
            for gamma, alpha in params:
                qaoa_layer(gamma, alpha)

        # Device setup
        dev = qml.device("default.qubit", wires=len(wires))

        # Cost function
        @qml.qnode(dev)
        def cost_function(params):
            circuit(params)
            return qml.expval(cost_h)

        # Optimizer setup
        opt = qml.AdagradOptimizer(stepsize=0.5)
        p = 5
        params = np.array([[0.5, 0.5]] * p, requires_grad=True)
        steps = 50
        tolerance = 0.005
        prev_cost = cost_function(params)
        iteration_logs = []

        # Optimization loop
        for i in range(steps):
            params = opt.step(cost_function, params)
            cost_val = cost_function(params)
            if (i + 1) % 5 == 0:
                iteration_logs.append(f"{i + 1:>2}: Cost = {cost_val:.4f}")
            if abs(cost_val - prev_cost) < tolerance:
                iteration_logs.append(f"Converged at iteration {i + 1}: Cost = {cost_val:.4f}")
                break
            prev_cost = cost_val

        optimized_params = params.tolist()

        # Sampling
        @qml.qnode(dev)
        def sample_output(params):
            circuit(params)
            return [qml.sample(qml.PauliZ(w)) for w in wires]

        bitstrings_raw = sample_output(params, shots=1000)
        bitstrings_transposed = np.transpose(bitstrings_raw)

        counts = {}
        for sample in bitstrings_transposed:
            bitstring = "".join([str(int((1 - s) / 2)) for s in sample])
            counts[bitstring] = counts.get(bitstring, 0) + 1

        most_freq_bitstring = max(counts, key=counts.get)
        coloring = [int(bit) for bit in most_freq_bitstring]
        node_colors = ['gold' if bit == 0 else 'lightblue' for bit in coloring]
        top_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20])

        NEON_BLUE = '#00f0ff'
        NEON_GREEN = '#00ffb3'
        NEON_ACCENT = '#11cdef'

        # --- Graph figure ---
        G = nx.Graph()
        G.add_nodes_from(wires)
        G.add_edges_from(edges)

        plt.style.use("dark_background")
        plt.figure(figsize=(4.5, 4.5), dpi=120)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G, pos,
            with_labels=True,
            node_color=node_colors,
            edge_color=NEON_ACCENT,
            node_size=720,
            font_size=14,
            font_color="black"
        )
        plt.title("Graph Partitioned by QAOA (MaxCut)", color=NEON_ACCENT)
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png', facecolor='black', bbox_inches='tight')
        plt.close()
        buf1.seek(0)

        # --- Histogram figure ---
        plt.figure(figsize=(max(len(top_counts) * 0.35, 3.6), 3.6), dpi=120)
        plt.bar(top_counts.keys(), top_counts.values(),
                color=NEON_GREEN, width=0.6, edgecolor=NEON_ACCENT)
        plt.xlabel('Bitstrings', color=NEON_BLUE)
        plt.ylabel('Counts', color=NEON_BLUE)
        plt.title('Most Frequently Sampled Bitstrings', color=NEON_ACCENT)
        plt.xticks(rotation=45, ha='right', color=NEON_BLUE)
        plt.yticks(color=NEON_BLUE)
        plt.tight_layout()
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', facecolor='black')
        plt.close()
        buf2.seek(0)

        # --- Convert to WebP & store in /tmp with a token ---
        token = uuid4().hex
        graph_path = TMP_DIR / f"qaoa-{token}-graph.webp"
        hist_path  = TMP_DIR / f"qaoa-{token}-hist.webp"

        Image.open(buf1).convert("RGB").save(str(graph_path), "WEBP", quality=78, method=6)
        Image.open(buf2).convert("RGB").save(str(hist_path),  "WEBP", quality=78, method=6)


        payload = {
            "token": token,
            "graph_url": f"/api/image/{token}/graph",
            "hist_url": f"/api/image/{token}/hist",
            "iterations": iteration_logs,
            "optimized_params": optimized_params,
            "most_freq_bitstring": most_freq_bitstring,
            "console_output": "Console Output:"
        }
        return no_store(jsonify(payload))
    except Exception as e:
        print("QAOA error:", e)
        return no_store(jsonify({"error": str(e)})), 500

@app.route("/api/image/<token>/<kind>")
def get_image(token, kind):
    kind = "graph" if kind == "graph" else "hist"
    path = TMP_DIR / f"qaoa-{token}-{kind}.webp"
    if not path.exists():
        return jsonify({"error": "Image expired or not found"}), 404
    resp = make_response(send_file(str(path), mimetype="image/webp"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == '__main__':
    app.run()

