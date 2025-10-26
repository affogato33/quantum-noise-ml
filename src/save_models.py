suppressor.save_models("my_models")
print("Models saved to 'my_models' directory")


new_suppressor = QuantumErrorSuppressor()
new_suppressor.load_models("my_models")


test_recommendation = new_suppressor.predict_mitigation_strategy(test_circuit)
print(f"Loaded model recommendation: {test_recommendation['recommended_strategy']}")
print("Model loaded successfully!")
