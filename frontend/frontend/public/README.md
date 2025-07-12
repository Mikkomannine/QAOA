# QAOA
```python
    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(alpha, mixer_h)

    def circuit(params):
        for w in wires:
            qml.Hadamard(wires=w)
        for gamma, alpha in params:
            qaoa_layer(gamma, alpha)

    dev = qml.device("default.qubit", wires=len(wires))

    @qml.qnode(dev)
    def cost_function(params):
        circuit(params)
        return qml.expval(cost_h)
```
## How it Works
