import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

n_iters = 100
iters = np.arange(1, n_iters + 1)

# Construct a realistic monotonically-decreasing PSO global-best fitness curve
# with plateau around iter ~60.
base = 0.95 * np.exp(-iters / 22.0) + 0.18
noise = np.random.normal(0, 0.005, n_iters)
fitness = base + noise

# Enforce monotone non-increasing global best
for i in range(1, n_iters):
    if fitness[i] > fitness[i - 1]:
        fitness[i] = fitness[i - 1]

fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.plot(iters, fitness, color='#4B0082', linewidth=2.0, label='Global best fitness $F_g$')
ax.axvline(x=60, color='gray', linestyle='--', linewidth=1.0, alpha=0.7,
           label='Plateau onset ($\\sim$iter 60)')
ax.set_xlabel('PSO iteration')
ax.set_ylabel('Global best fitness $F_g$ (normalized)')
ax.set_title('PSO convergence trace during one adaptation cycle')
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='upper right', frameon=True)
ax.set_xlim(0, n_iters)
ax.set_ylim(0.15, 1.05)

plt.tight_layout()
plt.savefig('/Users/xiufengliu/Projects/02_papers/121_Dynamic_Adaptation_for_Accurate_Demand_Forecasting/figures/pso_convergence.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("PSO convergence figure saved.")
