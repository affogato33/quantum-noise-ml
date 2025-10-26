#  visualizations
import matplotlib.pyplot as plt


strategies = list(recommendation['strategy_probabilities'].keys())
probs = list(recommendation['strategy_probabilities'].values())

plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange'] 
bars = plt.bar(strategies, probs, color=colors[:len(strategies)])
plt.title('Strategy Recommendation Probabilities')
plt.ylabel('Probability')
plt.xlabel('Strategy')
plt.xticks(rotation=45)


for bar, prob in zip(bars, probs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{prob:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("Visualization complete!")
