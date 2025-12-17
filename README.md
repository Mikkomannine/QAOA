# QAOA for Maxcut

Live Demo: [https://qaoa-maxcut.onrender.com/](https://qaoa-maxcut.onrender.com/)

# What is QAOA?

The Quantum Approximate Optimization Algorithm (QAOA) is a hybrid quantum-classical algorithm designed to solve combinatorial optimization problems such as MaxCut.

QAOA works by preparing a parameterized quantum state using alternating layers of quantum gates based on two Hamiltonians:

- A cost Hamiltonian that encodes the objective function.

- A mixer Hamiltonian that enables transitions between solutions.

The parameters are optimized using a classical algorithm to minimize the expectation value of the cost Hamiltonian. The final quantum state is measured, and the most frequently observed bitstring approximates the optimal solution.

QAOA is particularly useful for problems where classical solutions are hard to compute but can be encoded efficiently into quantum circuits. We’ll explore the underlying logic more deeply in the code example later on.

# Introduction to MaxCut Problem

The **MaxCut problem** is a classical graph optimization problem. Given a graph with a set of nodes (vertices) and edges (connections between nodes), the goal is to divide the nodes into two groups such that the number of edges between the groups is as large as possible.

In other words, we want to "cut" the graph in a way that maximizes the number of edges crossing between the two sides.

This problem is known to be **NP-hard**, meaning it becomes very difficult to solve exactly as the size of the graph increases. However, it has many applications in fields like physics, circuit design, and network analysis — which is why approximate quantum algorithms like QAOA are valuable tools for tackling it.

---

## Visual Example

Let's consider the graph shown below:

![MaxCut Example](/frontend/public/images/maxcut_example.png)

In this example:

- The nodes are colored into **two groups**:
  - **Yellow nodes**: 0, 2, 3
  - **Blue nodes**: 1, 4
- The **red curve** visually separates the two groups.

The goal is to count the number of edges that cross between the yellow and blue groups.

These **cut edges** are:
- (0,1)
- (1,2)
- (2,4)
- (3,4)
- (0,4)

There are 5 edges crossing the cut, giving a cut value of 5. A bitstring that produces this cut is 10110 (or equivalently, 01001, since the partition is symmetric)

This illustrates what the MaxCut problem is all about: **finding the best way to separate the graph into two groups** so that the number of crossing edges is maximized.

---

Code Walkthrough
---

## 1. Graph Generation

```python
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
```
A graph with 3 to 5 nodes is randomly generated. Each node is connected to at least one other node, and additional edges are added probabilistically. Variable **num_wires** is the number of qubits that each represent a node. 

The **MaxCut problem** aims to partition the nodes into two sets such that the number of edges between the sets is maximized.

---

## 2. Cost and Mixer Hamiltonians

```python
cost_h, mixer_h = qml.qaoa.maxcut(graph)
```
PennyLane provides utility functions to generate:

- **Cost Hamiltonian** $$H_C$$, which encodes the MaxCut objective:
$$
H_C = \sum_{(i,j)\in E} \frac{1}{2}(Z_i Z_j - I)
$$

Where:
- $$Z_i$$ is the **Pauli-Z** operator acting on qubit i.
- $${(i,j)\in E}$$ means the edge connects nodes i and j.
- $$I$$ is identity operator with eigenvalue of 1.
- $$\frac{1}{2}$$ is a rescaling factor, used to make each term contribute 0 or 1 depending on whether the edge is cut.

Each term in $$H_C$$ contributes:
- **0** if qubits i and j are in the **same group** (both 0 or both 1)
- **-1** if they are in **opposite groups** — meaning the edge is **cut**

So the **expected value** of $$H_C$$, given a quantum state, represents the **total number of edges cut** by that state's bitstring.


- **Mixer Hamiltonian** $$H_M$$, which drives transitions between basis states:
$$
H_M = \sum_{i\in nodes} X_i
$$

Where:
- $$X_i$$ is Pauli-X operator acting on qubit i.
- This allows the quantum state to **move across the solution landscape**, rather than being stuck in one configuration.

Together with the cost unitary, it enables **constructive interference** toward better solutions and **destructive interference** against poor ones.

---

## 3. QAOA Layer

```python
def qaoa_layer(gamma, alpha):
    qml.qaoa.cost_layer(gamma, cost_h)
    qml.qaoa.mixer_layer(alpha, mixer_h)
```
Each QAOA layer applies the unitaries with adjustable parameters based on the cost and mixer Hamiltonians:

$$
U(\gamma, \alpha) = e^{-i \alpha H_M} e^{-i \gamma H_C}
$$

These layers are applied $$p$$ times (depth of the algorithm).

---

## 4. Quantum Circuit

```python
    def circuit(params):
        for w in wires:
            qml.Hadamard(wires=w)
        for gamma, alpha in params:
            qaoa_layer(gamma, alpha)
```
![QAOA Circuit Diagram](images/circuit.png)

Applies Hadamard gates to all qubits to prepare the initial state:

$$
|\psi_0\rangle = \frac{1}{\sqrt{2^n}} \sum_{z \in \{0,1\}^n} |z\rangle
$$

The system begins in an equal superposition over all bitstrings.

The quantum state prepared by QAOA is defined as:
$$
|\psi(\boldsymbol{\gamma}, \boldsymbol{\alpha})\rangle = \prod_{k=1}^{p} e^{-i \alpha_k H_M} e^{-i \gamma_k H_C} |+\rangle^{\otimes n}
$$

Where:
- $$\prod_{k=1}^{p}$$ ordered application of QAOA layers from $$𝑘 = 1$$ to $$p$$
- $$p$$ is the number of QAOA layers.
- $$H_C$$ is the cost hamiltonian.
- $$H_M$$ is the mixer hamiltonian.
- $$\gamma_k, \alpha_k$$ are layer-specific variational parameters which we need to optimize using classical optimizer to minimize the expectation value of the cost Hamiltonian.
- $$|+\rangle^{\otimes n}$$ represents the equal superposition state over $$n$$ qubits.




---


## 5. Cost Function and Optimization

```python
p = 5
opt = qml.AdagradOptimizer(stepsize=0.5)
params = np.array([[0.5, 0.5]] * p, requires_grad=True)
```
The classical **Adagrad optimizer** is used to minimize the expected cost of the quantum circuit. The cost function is defined as the **expectation value of the cost Hamiltonian** $$H_C$$ with respect to the current quantum state $$|\psi(\gamma, \alpha)
\rangle$$:

```python
@qml.qnode(dev)
def cost_function(params):
    circuit(params)
    return qml.expval(cost_h)
```

The goal is to minimize:
$$
\langle \psi(\gamma, \alpha) | H_C | \psi(\gamma, \alpha) \rangle
$$

Where $$|\psi(\gamma, \alpha)\rangle$$ is the quantum state produced after applying $$p$$ layers with given parameters.

---
### How the Optimizer Works

The optimization loop in QAOA combines quantum circuit evaluation with classical gradient-based updates. Here's a breakdown:

---

#### Estimate Gradient (Parameter-Shift Rule)

Since we can't directly compute gradients on a quantum circuit, we use the **parameter-shift rule**:
$$
\frac{\partial \mathcal{L}(\theta)}{\partial \theta} \approx \frac{1}{2} \left[
\mathcal{L}(\theta + \frac{\pi}{2}) - \mathcal{L}(\theta - \frac{\pi}{2})
\right]
$$
- $$\mathcal{L}(\theta)$$ The loss function (cost hamiltonian)
- $$\theta$$ A variational parameter

---

#### Apply Optimizer Update (Adagrad)

Adagrad is an adaptive optimization algorithm that adjusts the learning rate individually for each parameter based on the accumulated history of its squared gradients, enabling more stable and efficient convergence. The parameter update rule is:

$$
\theta_i^{(t+1)} = \theta_i^{(t)} - \frac{\eta}{\sqrt{G_i^{(t)} + \epsilon}} \cdot g_i^{(t)}
$$

- $$G_i^{(t)}$$: gradient at step $$t$$
- $$G_i^{(t)}$$: accumulated squared gradients
- $$\eta$$: learning rate
- $$\epsilon$$: small constant for stability

This helps reduce the step size for frequently updated parameters, making learning more stable and adaptive.

---

#### Repeat

QAOA repeats the process:
- Estimates new cost
- Calculates gradients
- Updates parameters

This hybrid loop continues until the cost converges or a set number of iterations are complete.

---

## 6. Optimization Loop

```python
    steps = 50
    tolerance = 0.005
    prev_cost = cost_function(params)
    iteration_logs = []

    for i in range(steps):
        params = opt.step(cost_function, params)
        cost_val = cost_function(params)
        if abs(cost_val - prev_cost) < tolerance:
            break
        prev_cost = cost_val
```
The optimizer updates the parameters $$\gamma_k, \alpha_k$$ for each layer to find the minimum expectation value of the cost hamiltonian.

---

## 7. Sampling the Optimized Circuit

```python
@qml.qnode(dev)
def sample_output(params):
    circuit(params)
    return [qml.sample(qml.PauliZ(w)) for w in wires]

bitstrings_raw = sample_output(params, shots=1000)
```
After the parameters are optimized, we execute the circuit 1000 times using the final parameter values to obtain bitstrings. Each execution performs projective measurements in the Pauli-Z basis on all qubits, returning results of -1 or +1 for each wire (qubit).

The output (bitstrings_raw) is a huge array with 1000 columns and n_qubit rows.

---

## 8. Bitstring Conversion and Counting

```python
bitstrings_transposed = np.transpose(bitstrings_raw)

for sample in bitstrings_transposed:
    bitstring = "".join([str(int((1 - s) / 2)) for s in sample])
```
At first, we transpose the raw bitstring array. Now each row represents one complete bitstring.

Pauli-Z measurements return -1 or 1. The for-loop maps the values to binary:

- +1 $$\rightarrow$$ 0
- -1 $$\rightarrow$$ 1

Most frequently measured bitstring is interpreted as the **MaxCut solution**.

---

## 9. Visualization

```python
node_colors = ['gold' if bit == 0 else 'lightblue' for bit in coloring]
```
Nodes are colored according to their partition in the MaxCut.

---

## Summary

- **Quantum part** prepares and evolves a superposition over solutions using QAOA layers, and approximates the lowest energy state of the cost Hamiltonian by evaluating the expectation value of the cost Hamiltonian on the final state.
- **Classical part** optimizes parameters to minimize the cost Hamiltonian.
- **Result**: The most probable bitstring corresponds to the approximate solution to MaxCut.

---