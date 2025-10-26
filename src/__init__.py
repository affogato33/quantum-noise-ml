# Initialize the framework
suppressor = QuantumErrorSuppressor()
info = suppressor.get_suppressor_info()

print("Framework Information:")
for key, value in info.items():
    print(f"  {key}: {value}")

print("\nAvailable Strategies:")
strategies = suppressor.get_available_strategies()
for strategy in strategies:
    print(f"  - {strategy}")
