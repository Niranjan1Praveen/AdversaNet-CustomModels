import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('attack_success_rate.csv')

# Plot setup
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Model', y='Classification Flip Rate (%)', hue='Attack')

# Customization
plt.title('Attack Success Rate Comparison')
plt.ylabel('Classification Flip Rate (%)')
plt.ylim(0, 110)
plt.legend(title='Attack Type')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Show plot
plt.tight_layout()
plt.show()
