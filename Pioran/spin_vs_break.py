import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 1. Load the aggregated data adjust the path as necessary
df = pd.read_csv('master_pioran_results.csv')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(11, 7), dpi=150)

norm = mcolors.Normalize(vmin=df['r_high'].min(), vmax=df['r_high'].max())
cmap = plt.get_cmap('viridis')

# Map unique inclinations to specific marker shapes
# inc_markers = {50: 'o', 60: '^', 70: 's', 80: 'D'} # Adjust keys to match your actual inclination values
default_marker = 'o'

# 1. Define a jitter offset map for the x-axis based on R_high to prevent total overlap
unique_r = sorted(df['r_high'].unique())
jitter_offsets = {r: offset for r, offset in zip(unique_r, [-0.03, -0.01, 0.01, 0.03])}

# Group by both R_high and inclination to handle colors and markers simultaneously
for (r_val, inc_val), group in df.groupby(['r_high', 'inclination']):
    color = cmap(norm(r_val))
    # marker = inc_markers.get(inc_val, default_marker)
    
    # Apply the jitter to the spin coordinate
    jittered_spin = group['spin'] + jitter_offsets.get(r_val, 0)
    
    ax.errorbar(
        x=jittered_spin,
        y=group['f1_break'],
        yerr=group['f1_break_error'],
        fmt=default_marker,
        color=color,
        markersize=7,
        capsize=3,
        alpha=0.75,
        elinewidth=1.2,
        markeredgecolor='black',
        markeredgewidth=0.5,
        label=f"$R_{{high}}$={r_val}" 
    )

ax.set_title("Impact of Black Hole Spin on Break Frequency ($f_1$)", fontsize=15, pad=15, weight='bold')
ax.set_xlabel("Black Hole Spin ($a$)", fontsize=12, labelpad=10)
ax.set_ylabel("Break Frequency ($f_1$ break)", fontsize=12, labelpad=10)

ax.set_xlim(-1.1, 1.1)
ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])

# Clean up duplicate labels in legend while keeping the breakdown clear
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), title="Model Parameters", 
          bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9)

plt.tight_layout()
# Save the figure in a path you want
plt.savefig('MAD_data/spin_vs_f1_break.png', bbox_inches='tight')
plt.show()