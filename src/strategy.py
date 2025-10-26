#  ML recommendation
recommendation = suppressor.predict_mitigation_strategy(test_circuit)

print("Strategy Recommendation:")
print(f"  Recommended: {recommendation['recommended_strategy']}")
print(f"  Confidence: {recommendation['strategy_probabilities'][recommendation['recommended_strategy']]:.3f}")

print("\nAll Strategy Probabilities:")
for strategy, prob in recommendation['strategy_probabilities'].items():
    print(f"  {strategy}: {prob:.3f}")

print("\nTop 3 Strategies:")
for i, (strategy, confidence) in enumerate(recommendation['top_strategies'], 1):
    print(f"  {i}. {strategy}: {confidence:.3f}")
