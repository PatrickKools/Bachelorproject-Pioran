import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Laad de data
df = pd.read_csv('output_intensity/grid_hypothesis_results_MAD.csv')


plt.figure(figsize=(10, 6))
for Rh in df['Rh'].unique():
    sub_df = df[df['Rh'] == Rh]
    plt.plot(sub_df['inclination'], sub_df['centroid_retro']*0.1, '--', label=f'Retro (Rh={Rh})', alpha=0.7)
    plt.plot(sub_df['inclination'], sub_df['centroid_pro']*0.1, 'o-', label=f'Pro (Rh={Rh})')

plt.title('Average emission radius vs. Inclination per $R_h$', fontsize=13)
plt.xlabel('Inclination ($i$ in degrees)', fontsize=11)
plt.ylabel('Average emission radius ($r_g$)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('average_radius_intensity.png', dpi=300)
plt.show()