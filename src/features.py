#simple test circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

print("Test Circuit:")
print(f"  Qubits: {circuit.num_qubits}")
print(f"  Depth: {circuit.depth()}")
print(f"  Size: {circuit.size()}")

# Display the circuit
print("\nCircuit Diagram:")
print(circuit.draw())


from hybrid_quantum_ml.features import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_features(circuit)

print("Extracted Features:")
print(f"  Total gates: {features['total_gates']}")
print(f"  Two-qubit ratio: {features['two_qubit_ratio']:.3f}")
print(f"  Connectivity density: {features['connectivity_density']:.3f}")
print(f"  Single-qubit ratio: {features['single_qubit_ratio']:.3f}")
print(f"  Entanglement ratio: {features['entanglement_ratio']:.3f}")

print("\nFeature extracted")
