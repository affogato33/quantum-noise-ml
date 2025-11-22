try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("scikit-optimize not available. Install with: pip install scikit-optimize")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

print("\nClassical State-of-the-Art Comparison")
print("="*70)

search_space_bench = {
    "C": {"type": "continuous", "low": 0.1, "high": 10.0, "num_points": 8},
    "gamma": {"type": "continuous", "low": 0.001, "high": 1.0, "num_points": 8}
}

objective_bench = SklearnObjective(SVC, X_train_w, y_train_w, X_test_w, y_test_w)
objective_bench.set_search_space(search_space_bench)

num_configs_bench = 8 * 8
max_evaluations = 60

def evaluate_config_sklearn(config_dict):
    try:
        model = SVC(**config_dict)
        model.fit(X_train_w, y_train_w)
        return -model.score(X_test_w, y_test_w)
    except:
        return 1.0

benchmark_results = {}

print("\n1. QHBO (Quantum Hyperparameter Bayesian Optimization)")
backend_bench = QiskitBackend(num_qubits=6, noise_model=None)
optimizer_bench = QHBOOptimizer(
    objective=objective_bench,
    backend=backend_bench,
    max_iterations=25,
    num_samples_per_iteration=None,
    verbose=False,
    show_quantum_details=False,
    learning_rate=0.4,
    entropy_regularization=0.03
)

start_time = time.time()
results_qhbo_bench = optimizer_bench.optimize()
qhbo_time = time.time() - start_time
qhbo_evals = results_qhbo_bench['num_iterations'] * optimizer_bench.num_samples_per_iteration

benchmark_results['QHBO'] = {
    'score': results_qhbo_bench['best_score'],
    'time': qhbo_time,
    'evaluations': qhbo_evals
}

print(f"  Best score: {results_qhbo_bench['best_score']:.4f}")
print(f"  Evaluations: {qhbo_evals}")
print(f"  Time: {qhbo_time:.2f}s")

if SKOPT_AVAILABLE:
    print("\n2. Gaussian Process Bayesian Optimization (scikit-optimize)")
    
    space_skopt = [
        Real(0.1, 10.0, name='C'),
        Real(0.001, 1.0, name='gamma')
    ]
    
    def objective_skopt(params):
        config = {'C': params[0], 'gamma': params[1]}
        return evaluate_config_sklearn(config)
    
    start_time = time.time()
    result_skopt = gp_minimize(
        objective_skopt,
        space_skopt,
        n_calls=max_evaluations,
        random_state=42,
        n_jobs=1
    )
    skopt_time = time.time() - start_time
    
    best_config_skopt = {'C': result_skopt.x[0], 'gamma': result_skopt.x[1]}
    best_score_skopt = -result_skopt.fun
    
    benchmark_results['GP-BO'] = {
        'score': best_score_skopt,
        'time': skopt_time,
        'evaluations': max_evaluations
    }
    
    print(f"  Best score: {best_score_skopt:.4f}")
    print(f"  Evaluations: {max_evaluations}")
    print(f"  Time: {skopt_time:.2f}s")

if OPTUNA_AVAILABLE:
    print("\n3. Tree-structured Parzen Estimator (Optuna)")
    
    def objective_optuna(trial):
        C = trial.suggest_float('C', 0.1, 10.0, log=True)
        gamma = trial.suggest_float('gamma', 0.001, 1.0, log=True)
        config = {'C': C, 'gamma': gamma}
        return -evaluate_config_sklearn(config)
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    
    start_time = time.time()
    study.optimize(objective_optuna, n_trials=max_evaluations, show_progress_bar=False)
    optuna_time = time.time() - start_time
    
    best_score_optuna = -study.best_value
    best_config_optuna = study.best_params
    
    benchmark_results['TPE (Optuna)'] = {
        'score': best_score_optuna,
        'time': optuna_time,
        'evaluations': max_evaluations
    }
    
    print(f"  Best score: {best_score_optuna:.4f}")
    print(f"  Evaluations: {max_evaluations}")
    print(f"  Time: {optuna_time:.2f}s")

print("\n4. Random Search")
def random_search_bench(objective, search_space, n_evaluations):
    best_score = -np.inf
    best_config = None
    for _ in range(n_evaluations):
        config = {}
        for param_name, param_def in search_space.items():
            if param_def["type"] == "continuous":
                config[param_name] = np.random.uniform(param_def["low"], param_def["high"])
            else:
                config[param_name] = np.random.choice(param_def["values"])
        score = -evaluate_config_sklearn(config)
        if score > best_score:
            best_score = score
            best_config = config
    return best_score, best_config

np.random.seed(42)
start_time = time.time()
random_score, random_config = random_search_bench(evaluate_config_sklearn, search_space_bench, max_evaluations)
random_time = time.time() - start_time

benchmark_results['Random Search'] = {
    'score': random_score,
    'time': random_time,
    'evaluations': max_evaluations
}

print(f"  Best score: {random_score:.4f}")
print(f"  Evaluations: {max_evaluations}")
print(f"  Time: {random_time:.2f}s")

print("\n" + "="*70)
print("Benchmark Summary:")
print(f"{'Method':<20} {'Score':<12} {'Time (s)':<12} {'Evaluations':<12} {'Score/Time':<12}")
print("-"*70)

baseline_score = benchmark_results.get('GP-BO', benchmark_results.get('Random Search', {}))['score']

for method, results in benchmark_results.items():
    score = results['score']
    time_val = results['time']
    evals = results['evaluations']
    score_per_time = score / time_val if time_val > 0 else 0
    improvement = score - baseline_score
    
    print(f"{method:<20} {score:<12.4f} {time_val:<12.2f} {evals:<12} {score_per_time:<12.4f}")

if len(benchmark_results) > 1:
    best_method = max(benchmark_results.items(), key=lambda x: x[1]['score'])
    print(f"\nBest performing method: {best_method[0]} (score: {best_method[1]['score']:.4f})")
    
    if 'QHBO' in benchmark_results:
        qhbo_score = benchmark_results['QHBO']['score']
        if qhbo_score >= baseline_score:
            advantage = qhbo_score - baseline_score
            print(f"QHBO advantage over baseline: +{advantage:.4f} ({advantage/baseline_score*100:.2f}% relative)")
        else:
            deficit = baseline_score - qhbo_score
            print(f"QHBO deficit vs baseline: -{deficit:.4f} ({deficit/baseline_score*100:.2f}% relative)")
            print("Note: Quantum advantage may emerge on larger problems or with different hyperparameters")
