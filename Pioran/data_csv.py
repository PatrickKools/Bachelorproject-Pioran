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
        
        
    for run_folder in os.listdir(spin_path):
        run_path = os.path.join(spin_path, run_folder)
        

        # matches 'Rh' followed by digits, and 'i' followed by digits
        match = re.search(r'Rh(\d+)_i(\d+)', run_folder)
        
        if match:
            r_high = int(match.group(1))
            inclination = int(match.group(2))
            
            # Path to the results subfolder
            results_dir = os.path.join(run_path, 'info')
            
            if os.path.exists(results_dir):
                # paths to the two potential summary files
                csv_path = os.path.join(results_dir, 'post_summary.csv')
                
                parsed = False
                
                # make csv file
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
                
                
                # add logic
                if parsed:
                    all_results.append({
                        'spin_folder': spin_folder,
                        'spin': spin_val,
                        'r_high': r_high,
                        'inclination': inclination,
                        
                        #
                        'alpha': alpha2_val,            
                        'alpha_error': alpha2_err,
                        'nu_high': f1_val,              
                        'nu_high_error': f1_err,
                        
                        
                        'alpha1': alpha1_val,           
                        'alpha1_error': alpha1_err,
                        'alpha2': alpha2_val,           
                        'alpha2_error': alpha2_err,
                        'f1_break': f1_val,             
                        'f1_break_error': f1_err,
                        'variance': variance_val,       
                        'nu_shape': nu_val,             
                        'mu_mean_flux': mu_val          
                    })

# Convert to a master DataFrame
df = pd.DataFrame(all_results)
df.to_csv('master_pioran_SANE_results.csv', index=False)
print(f"Compiled {len(df)} runs successfully.")



