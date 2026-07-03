import os
import re
import pandas as pd

# Path to the directory containing your spin folders (e.g., 'a0', 'a+1o2', etc.)
base_dir = "./inference/SANE"

spin_mapping = {
    'a0': 0.0,
    'a+1o2': 0.5,
    'a-1o2': -0.5,
    'a+15o16': 15/16,
    'a-15o16': -15/16
}

all_results = []

# Loop through each spin directory
for spin_folder, spin_val in spin_mapping.items():
    spin_path = os.path.join(base_dir, spin_folder)
    
    if not os.path.exists(spin_path):
        continue
        
        
    # List the parameter folders inside the spin directory (e.g., lc_Rh40_i50.npz_single)
    for run_folder in os.listdir(spin_path):
        run_path = os.path.join(spin_path, run_folder)
        
        # Parse Rh and inclination using regex
        # Matches 'Rh' followed by digits, and 'i' followed by digits
        match = re.search(r'Rh(\d+)_i(\d+)', run_folder)
        
        if match:
            r_high = int(match.group(1))
            inclination = int(match.group(2))
            
            # Path to the results subfolder
            results_dir = os.path.join(run_path, 'info')
            
            if os.path.exists(results_dir):
                # Explicit paths to the two potential summary files
                csv_path = os.path.join(results_dir, 'post_summary.csv')
                json_path = os.path.join(results_dir, 'results.json')
                
                parsed = False
                
                # --- STRATEGY 1: PARSE THE CSV SUMMARY (Preferred) ---
                if os.path.exists(csv_path):
                    try:
                        df_csv = pd.read_csv(csv_path)
                        
                        # Extract medians and standard deviations using exact Unicode column headers
                        alpha1_val = df_csv['α₁_median'].values[0]
                        alpha1_err = df_csv['α₁_stdev'].values[0]
                        
                        f1_val     = df_csv['f₁_median'].values[0]
                        f1_err     = df_csv['f₁_stdev'].values[0]
                        
                        alpha2_val = df_csv['α₂_median'].values[0]
                        alpha2_err = df_csv['α₂_stdev'].values[0]
                        
                        variance_val = df_csv['variance_median'].values[0]
                        variance_err = df_csv['variance_stdev'].values[0]
                        
                        nu_val     = df_csv['ν_median'].values[0]
                        nu_err     = df_csv['ν_stdev'].values[0]
                        
                        mu_val     = df_csv['μ_median'].values[0]
                        mu_err     = df_csv['μ_stdev'].values[0]
                        
                        parsed = True
                    except Exception as e:
                        print(f"Warning: Could not parse CSV at {csv_path}. Error: {e}")
                
                # --- STRATEGY 2: FALLBACK TO JSON (If CSV is missing or fails) ---
                if not parsed and os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            js = json.load(f)
                        
                        posterior = js['posterior']
                        
                        # PIORAN parameter order: [α₁, f₁, α₂, variance, ν, μ]
                        alpha1_val = posterior['median'][0]
                        alpha1_err = posterior['stdev'][0]
                        
                        f1_val     = posterior['median'][1]
                        f1_err     = posterior['stdev'][1]
                        
                        alpha2_val = posterior['median'][2]
                        alpha2_err = posterior['stdev'][2]
                        
                        variance_val = posterior['median'][3]
                        variance_err = posterior['stdev'][3]
                        
                        nu_val     = posterior['median'][4]
                        nu_err     = posterior['stdev'][4]
                        
                        mu_val     = posterior['median'][5]
                        mu_err     = posterior['stdev'][5]
                        
                        parsed = True
                    except Exception as e:
                        print(f"Warning: Could not parse JSON at {json_path}. Error: {e}")
                
                # --- APPEND LOGIC ---
                if parsed:
                    all_results.append({
                        'spin_folder': spin_folder,
                        'spin': spin_val,
                        'r_high': r_high,
                        'inclination': inclination,
                        
                        # Retaining your original placeholder naming conventions:
                        'alpha': alpha2_val,            # Maps high-frequency slope to alpha
                        'alpha_error': alpha2_err,
                        'nu_high': f1_val,              # Maps break frequency to nu_high
                        'nu_high_error': f1_err,
                        
                        # Comprehensive parameters (highly useful for advanced plotting):
                        'alpha1': alpha1_val,           # Low-frequency power-law slope
                        'alpha1_error': alpha1_err,
                        'alpha2': alpha2_val,           # High-frequency power-law slope
                        'alpha2_error': alpha2_err,
                        'f1_break': f1_val,             # Bending/break frequency
                        'f1_break_error': f1_err,
                        'variance': variance_val,       # Amplitude variance of the GP
                        'nu_shape': nu_val,             # Smoothness/shape variable
                        'mu_mean_flux': mu_val          # Mean flux level of the lightcurve
                    })

# Convert to a master DataFrame
df = pd.DataFrame(all_results)
print(df[['variance']].head())  # Quick check to ensure variance columns are populated
df.to_csv('master_pioran_SANE_results.csv', index=False)
print(f"Compiled {len(df)} runs successfully.")



