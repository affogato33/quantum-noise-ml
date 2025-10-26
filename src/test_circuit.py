# Create a test circuit
test_circuit = QuantumCircuit(3, 3)
test_circuit.h(0)
test_circuit.cx(0, 1)
test_circuit.cx(1, 2)
test_circuit.measure_all()

print(f"Test circuit: {test_circuit.num_qubits} qubits, depth {test_circuit.depth()}")
print("\nTest Circuit Diagram:")
print(test_circuit.draw())
