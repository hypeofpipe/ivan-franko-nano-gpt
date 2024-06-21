import numpy as np
import matplotlib.pyplot as plt

def layer_norm(x, epsilon=1e-5, gamma=1.0, beta=0.0):
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + epsilon)
    return gamma * normalized + beta

# Generate sample data
np.random.seed(42)
data = np.random.randn(5, 100)  # 5 samples, 100 features each

# Apply layer normalization
normalized_data = layer_norm(data)
final_output = layer_norm(data, gamma=1.5, beta=0.5)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(12, 15))
fig.suptitle('Layer Normalization Visualization', fontsize=16)

axs[0].set_title('Original Data')
axs[0].boxplot(data.T)
axs[0].set_xlabel('Sample')
axs[0].set_ylabel('Value')

axs[1].set_title('Normalized Data (γ=1, β=0)')
axs[1].boxplot(normalized_data.T)
axs[1].set_xlabel('Sample')
axs[1].set_ylabel('Value')

axs[2].set_title('Final Output (γ=1.5, β=0.5)')
axs[2].boxplot(final_output.T)
axs[2].set_xlabel('Sample')
axs[2].set_ylabel('Value')

plt.tight_layout()
plt.show()