try:
    result = suppressor.execute_with_mitigation(test_circuit, strategy='ZNE', shots=1024)
    
    print("Execution Results:")
    print("Result keys:", list(result.keys()))
    
   
    expectation_value = "N/A"
    if 'mitigated_result' in result and isinstance(result['mitigated_result'], dict):
        if 'expectation_value' in result['mitigated_result']:
            expectation_value = result['mitigated_result']['expectation_value']
    
    print(f"  Expectation value: {expectation_value}")
    print(f"  Strategy used: ZNE")
    print(f"  Shots: 1024")
    
    print("\nMitigation execution complete")
    
except Exception as e:
    print(f"Execution failed: {e}")
    print("Trying basic execution...")
    
   
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    job = backend.run(test_circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
 
    expectation_value = 0.0
    total_shots = sum(counts.values())
    for state, count in counts.items():
        num_ones = state.count('1')
        parity = 1 if num_ones % 2 == 0 else -1
        expectation_value += count * parity
    expectation_value /= total_shots
    
    print(f"execution result: {expectation_value:.3f}")
    print(" execution complete")

# Compare all strategies
comparison = suppressor.compare_strategies(test_circuit, shots=1024)

print("Strategy Comparison:")
print(f"  Best strategy: {comparison.get('best_strategy', 'Unknown')}")
print(f"  Available strategies: {comparison.get('strategy_count', 0)}")

if 'all_results' in comparison:
    print("\nStrategy Results:")
    for strategy, result in comparison['all_results'].items():
        if isinstance(result, dict):
            
            exp_val = "N/A"
            for key in ['expectation_value', 'mitigated_result']:
                if key in result:
                    if isinstance(result[key], dict) and 'expectation_value' in result[key]:
                        exp_val = result[key]['expectation_value']
                    elif isinstance(result[key], (int, float)):
                        exp_val = result[key]
                    break
            
            print(f"  {strategy}: {exp_val}")
        else:
            print(f"  {strategy}: {result}")

# which strategies worked
working_strategies = []
if 'all_results' in comparison:
    for strategy, result in comparison['all_results'].items():
        if isinstance(result, dict) and ('expectation_value' in result or 'mitigated_result' in result):
            working_strategies.append(strategy)

print(f"\nWorking strategies: {working_strategies}")
print(f"Failed strategies: {[s for s in comparison.get('all_results', {}).keys() if s not in working_strategies]}")

print("\nStrategy comparison complete")

# Test batch execution with ZNE strategy 
test_circuits = create_training_circuits(5)

print("Testing batch execution with ZNE strategy")
batch_results = []

for i, circuit in enumerate(test_circuits):
    print(f"Processing circuit {i+1}/{len(test_circuits)}...")
    try:
        result = suppressor.execute_with_mitigation(circuit, strategy='ZNE', shots=512)
        batch_results.append(result)
        print(f" Circuit {i+1} successful")
    except Exception as e:
        print(f" Circuit {i+1} failed: {e}")
        batch_results.append({'error': str(e)})

successful = sum(1 for r in batch_results if 'error' not in r)
print(f"\nBatch execution: {successful}/{len(test_circuits)} successful")

if successful > 0:
    print("\nBatch Results Summary:")
    for i, result in enumerate(batch_results):
        if 'error' not in result:
          
            exp_val = "N/A"
            if 'expectation_value' in result:
                exp_val = result['expectation_value']
            elif 'mitigated_result' in result and isinstance(result['mitigated_result'], dict):
                if 'expectation_value' in result['mitigated_result']:
                    exp_val = result['mitigated_result']['expectation_value']
            
            print(f"  Circuit {i+1}: {exp_val}")

print("\nBatch execution test complete!")

# performance analysis
performance = suppressor.get_performance_analysis()

print("Performance Analysis:")
print(f"  Total executions: {performance.get('total_executions', 0)}")

if 'strategy_performance' in performance:
    print("\nStrategy Performance:")
    for strategy, perf in performance['strategy_performance'].items():
        print(f"  {strategy}: {perf['count']} uses, avg improvement: {perf.get('avg_expectation_improvement', 0):.3f}")

print("\nPerformance analysis complete")
