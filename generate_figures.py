"""Generate all paper figures from benchmark results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('results/figures', exist_ok=True)

# ============================================================
# Fig 1: Cross-domain main results bar chart
# ============================================================
domains = ['Acoustic\nMetamaterial', 'Airfoil\nAerodynamics', 'Concrete\nMaterials']
methods = ['LLM-Guided', 'Heuristic', 'Designer-Only', 'GA', 'BO', 'Random']
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#795548']

data = {
    'Acoustic\nMetamaterial': [0.894, 0.892, 0.811, 0.374, 0.359, 0.364],
    'Airfoil\nAerodynamics':  [0.534, 0.408, 0.533, 0.360, 0.920, 0.314],
    'Concrete\nMaterials':    [0.898, 0.921, 0.957, 1.368, 1.368, 1.269],
}

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(domains))
width = 0.13

for i, method in enumerate(methods):
    vals = [data[d][i] for d in domains]
    bars = ax.bar(x + i*width, vals, width, label=method, color=colors[i])
    for bar, val in zip(bars, vals):
        if val > 1.3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   'OOD!', ha='center', fontsize=7, color='red', fontweight='bold')

ax.set_ylabel('Normalized Objective (higher is better)', fontsize=12)
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(domains, fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_title('Cross-Domain Inverse Design Performance', fontsize=14, fontweight='bold')

ax.annotate('OOD false positives:\nsurrogate extrapolation error',
            xy=(2.3, 1.37), fontsize=9, color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', alpha=0.8))

plt.tight_layout()
plt.savefig('results/figures/fig1_main_results.png', dpi=300)
plt.close()
print('Fig 1 saved')

# ============================================================
# Fig 2: Concrete OOD Analysis
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

methods_conc = ['GA', 'BO', 'Random', 'Designer', 'Heuristic', 'LLM']
conc_vals = [113.0, 113.0, 104.8, 79.0, 76.1, 74.2]
conc_colors = ['#F44336','#F44336','#FF9800','#4CAF50','#2196F3','#2196F3']
bars = ax1.bar(methods_conc, conc_vals, color=conc_colors)
ax1.axhline(y=82.6, color='black', linestyle='--', linewidth=2, label='Training max (82.6 MPa)')
ax1.set_ylabel('Predicted Strength (MPa)', fontsize=12)
ax1.set_title('Concrete: Surrogate Predictions', fontsize=13)
ax1.legend(fontsize=10)
for bar, val in zip(bars, conc_vals):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.0f}',
            ha='center', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

ax2.barh(['GA', 'BO', 'Random'], [30.4, 30.4, 22.2], color='#F44336', label='OOD gap (+)')
ax2.barh(['LLM', 'Heuristic', 'Designer'], [-8.4, -6.5, -3.6], color='#4CAF50', label='In-dist (-)')
ax2.axvline(x=0, color='black')
ax2.set_xlabel('Deviation from Training Max (MPa)', fontsize=12)
ax2.set_title('OOD Gap Analysis', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/fig2_concrete_ood.png', dpi=300)
plt.close()
print('Fig 2 saved')

# ============================================================
# Fig 3: Ablation -- Full vs Minimal Prompt
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

prompts = ['Full DomainSpec', 'Minimal\n(no context)']
rounds_data = [0, 2]
avgs_data = [0.8924, 0.8956]

ax1.bar(prompts, rounds_data, color=['#4CAF50', '#FF9800'])
ax1.set_ylabel('Rounds to Convergence', fontsize=12)
ax1.set_title('Convergence Efficiency', fontsize=13)
for i, r in enumerate(rounds_data):
    ax1.text(i, r+0.1, f'{r} rounds\n(immediate pass)' if r==0 else f'{r} rounds\n(never converged)',
            ha='center', fontsize=11)

ax2.bar(prompts, avgs_data, color=['#4CAF50', '#FF9800'])
ax2.set_ylabel('Final Avg Absorption', fontsize=12)
ax2.set_title('Final Performance', fontsize=13)
for i, v in enumerate(avgs_data):
    ax2.text(i, v+0.001, f'{v:.4f}', ha='center', fontsize=11)

fig.suptitle('Ablation A3: DomainSpec Prompt Context Matters', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/fig3_ablation_prompt.png', dpi=300)
plt.close()
print('Fig 3 saved')

# ============================================================
# Fig 4: Acoustic domain -- method comparison
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
ac_methods = ['LLM-Guided', 'Heuristic', 'Designer-Only', 'BO', 'Random']
ac_vals = [0.894, 0.892, 0.811, 0.359, 0.364]
ac_colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#795548']
bars = ax.bar(ac_methods, ac_vals, color=ac_colors)
for bar, val in zip(bars, ac_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}',
            ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Absorption', fontsize=12)
ax.set_title('Acoustic Metamaterial: Method Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/fig4_acoustic_detail.png', dpi=300)
plt.close()
print('Fig 4 saved')

# ============================================================
# Fig 5: Refinement trace
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
rounds = [1, 2, 3, 4, 5]
acoustic_trace = [0.892, 0.891, 0.874, 0.894, 0.888]
ax.plot(rounds, acoustic_trace, 'o-', color='#2196F3', linewidth=2, markersize=10, label='LLM-Guided (acoustic)')
ax.set_xlabel('Refinement Round', fontsize=12)
ax.set_ylabel('Average Absorption', fontsize=12)
ax.set_title('LLM-Guided Refinement Trace', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/fig5_refinement_trace.png', dpi=300)
plt.close()
print('Fig 5 saved')

# ============================================================
# Fig 6: Contribution waterfall
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
components = ['Random\n(baseline)', '+ Gradient\nOptimization', '+ Multi-start', '+ Critic\n(Heuristic)', '+ LLM\nReasoning']
deltas = [0.364, 0.447, 0.000, 0.081, 0.002]
bottom = [0, 0.364, 0.811, 0.811, 0.892]
colors_wf = ['#795548', '#FF9800', '#FF9800', '#4CAF50', '#2196F3']
bars = ax.bar(components, deltas, bottom=bottom, color=colors_wf)
ax.set_ylabel('Average Absorption', fontsize=12)
ax.set_title('Component Contribution Analysis (Acoustic)', fontsize=14, fontweight='bold')
final_values = [0.364, 0.811, 0.811, 0.892, 0.894]
for i, (bar, v) in enumerate(zip(bars, final_values)):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/fig6_contribution_waterfall.png', dpi=300)
plt.close()
print('Fig 6 saved')

print()
print('All 6 figures saved to results/figures/')
