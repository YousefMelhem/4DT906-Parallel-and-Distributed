import matplotlib.pyplot as plt
import numpy as np

# Performance data
implementations = ['Naive', 'Basic Tiled', 'Optimized']
gflops = [1292, 1775, 6120]
times = [1.66, 1.38, 0.42]
speedups = [1.0, 1.4, 4.9]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# GFLOPS plot
bars1 = ax1.bar(implementations, gflops, color=['#FF9999', '#66B2FF', '#99FF99'])
ax1.set_ylabel('GFLOPS')
ax1.set_title('Performance Comparison (GFLOPS)')
ax1.grid(True, linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.0f}',
             ha='center', va='bottom')

# Execution Time plot
bars2 = ax2.bar(implementations, times, color=['#FF9999', '#66B2FF', '#99FF99'])
ax2.set_ylabel('Time (ms)')
ax2.set_title('Execution Time Comparison')
ax2.grid(True, linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

# Adjust layout and save
plt.tight_layout()
plt.savefig('plots/performance_comparison.png', dpi=300, bbox_inches='tight')

# Create speedup comparison plot
plt.figure(figsize=(10, 6))
bars3 = plt.bar(implementations, speedups, color=['#FF9999', '#66B2FF', '#99FF99'])
plt.ylabel('Speedup (vs Naive)')
plt.title('Speedup Comparison')
plt.grid(True, linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars3:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}x',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('plots/speedup_comparison.png', dpi=300, bbox_inches='tight')

# Create memory hierarchy utilization plot
memory_data = {
    'Global Memory Access': [3, 2, 1],  # 3=High, 2=Medium, 1=Low
    'Shared Memory Usage': [1, 2, 3],   # 1=None, 2=Basic, 3=Optimized
    'Register Usage': [1, 1, 3]         # 1=Low, 2=Medium, 3=High
}

plt.figure(figsize=(12, 6))
x = np.arange(len(implementations))
width = 0.25

for i, (key, values) in enumerate(memory_data.items()):
    plt.bar(x + i*width, values, width, label=key)

plt.ylabel('Utilization Level')
plt.title('Memory Hierarchy Utilization')
plt.xticks(x + width, implementations)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('plots/memory_utilization.png', dpi=300, bbox_inches='tight') 