# Create training circuits
def create_training_circuits(num_circuits=50): 
    circuits = []
    for i in range(num_circuits):
       
        num_qubits = np.random.randint(2, 6)
        circuit = QuantumCircuit(num_qubits, num_qubits)
        
       
        num_gates = np.random.randint(5, 20)
        for _ in range(num_gates):
            gate_type = np.random.choice(['h', 'x', 'y', 'z', 's', 't', 'cx', 'cy', 'cz'])
            
            if gate_type in ['h', 'x', 'y', 'z', 's', 't']:
                qubit = np.random.randint(num_qubits)
                getattr(circuit, gate_type)(qubit)
            else:  # Two-qubit gates
                control = np.random.randint(num_qubits)
                target = np.random.randint(num_qubits)
                if control != target:
                    getattr(circuit, gate_type)(control, target)
        
        circuit.measure_all()
        circuits.append(circuit)
    
    return circuits

#  training data
training_circuits = create_training_circuits(50)
print(f"Created {len(training_circuits)} training circuits")



def generate_improved_training_data(circuits):
   
    noise_scores = []
    optimal_strategies = []
    
    for circuit in circuits:
       
        depth = circuit.depth()
        num_qubits = circuit.num_qubits
        size = circuit.size()
        
       
        gate_counts = {}
        for instruction, _, _ in circuit.data:
            gate_name = instruction.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        
        # Calculate complexity metrics
        two_qubit_ratio = gate_counts.get('cx', 0) / size if size > 0 else 0
        t_gate_ratio = gate_counts.get('t', 0) / size if size > 0 else 0
        h_gate_ratio = gate_counts.get('h', 0) / size if size > 0 else 0
        
       
        noise_score = (
            depth * 0.15 +                    # Depth penalty
            num_qubits * 0.08 +               # Qubit count penalty
            two_qubit_ratio * 0.4 +           # Two-qubit gate penalty
            t_gate_ratio * 0.3 +              # T-gate penalty
            h_gate_ratio * 0.1 +              # H-gate penalty
            np.random.normal(0, 0.05)         # Small random noise
        )
        
        
        noise_score = max(0.1, noise_score)
        noise_scores.append(noise_score)
        
      
        if noise_score > 0.8 and two_qubit_ratio > 0.3:
            strategy = 'ZNE'
        elif noise_score > 0.6 and num_qubits > 4:
            strategy = 'MEM'
        elif noise_score > 0.4 and t_gate_ratio > 0.2:
            strategy = 'PEC'
        else:
            strategy = 'CDR'
        
        optimal_strategies.append(strategy)
    
    return noise_scores, optimal_strategies

#  improved training labels
noise_scores, optimal_strategies = generate_improved_training_data(training_circuits)

print("Improved Training data generated:")
print(f"  Noise scores range: {min(noise_scores):.3f} to {max(noise_scores):.3f}")
print(f"  Strategy distribution: {dict(zip(*np.unique(optimal_strategies, return_counts=True)))}")


def train_robust_models():
   
    try:
        print("Training ML models with improved data.")
        training_results = suppressor.train(training_circuits, noise_scores, optimal_strategies)
        
        print("Training completed!")
        print(f"Results: {list(training_results.keys())}")
        
        
        if 'noise_predictor' in training_results and 'strategy_selector' in training_results:
            noise_r2 = training_results['noise_predictor'].get('test_r2', 0)
            strategy_acc = training_results['strategy_selector'].get('test_accuracy', 0)
            
            print(f"\nModel Performance:")
            print(f"  Noise Predictor R²: {noise_r2:.3f}")
            print(f"  Strategy Selector Accuracy: {strategy_acc:.3f}")
            
            if noise_r2 < 0.3:
                print("Warning: Low R² score. Consider more training data or feature engineering.")
            if strategy_acc < 0.7:
                print("Warning: Low accuracy. Consider more diverse training data.")
        
        return training_results
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Trying with simpler model.")
        return None


training_results = train_robust_models()

# Model validation
def validate_models():
  
    if training_results is None:
        print(" No training results to validate")
        return
    
    print("\nMODEL VALIDATION ")
    
    # Validate noise predictor
    noise_results = training_results.get('noise_predictor', {})
    noise_r2 = noise_results.get('test_r2', 0)
    
    print(f"Noise Predictor:")
    print(f"  R² Score: {noise_r2:.3f}")
    if noise_r2 > 0.7:
        print("  Excellent performance")
    elif noise_r2 > 0.5:
        print("  Good performance")
    elif noise_r2 > 0.3:
        print("  Moderate performance - consider more training data")
    else:
        print(" Poor performance - needs improvement")
    
    # Validate strategy selector
    strategy_results = training_results.get('strategy_selector', {})
    strategy_acc = strategy_results.get('test_accuracy', 0)
    
    print(f"\nStrategy Selector:")
    print(f"  Accuracy: {strategy_acc:.3f}")
    if strategy_acc > 0.8:
        print("  Excellent performance")
    elif strategy_acc > 0.7:
        print("  Good performance")
    elif strategy_acc > 0.6:
        print("  Moderate performance - consider more training data")
    else:
        print("  Poor performance - needs improvement")
    
    # Recommendations
    print(f"\n RECOMMENDATIONS ")
    if noise_r2 < 0.5 or strategy_acc < 0.7:
        print("1. Increase training data size (try 100+ circuits)")
        print("2. Add more diverse circuit types")
        print("3. Tune hyperparameters")
        print("4. Consider feature engineering")
    
    print("Model validation complete!")

# Run validation
validate_models()
