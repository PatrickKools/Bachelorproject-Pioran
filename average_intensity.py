import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  
from astropy.io import fits

def calculate_radial_profiles(start_num=1500, end_num=1550, base_dir="/zfs/helios/filer0/oporth/share/MAD-rhovfix/a+15o16/fits"):
 
    all_profiles = []
    bin_centers = None
    processed_files = 0

    print(f"Starting batch processing of files from {start_num} to {end_num}...")
    print(f"Searching in directory: {os.path.abspath(base_dir)}\n")

    for idx in range(start_num, end_num + 1):
        # Construct the filename based on the current index, Rh, and inclination
        filename = f"img_s{idx}_Rh10_i30.fits"
        filepath = os.path.join(base_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  [Skipping] {filename} not found.")
            continue
            
        print(f"  [Processing] {filename}...")
        
        with fits.open(filepath) as hdul:
            data_cube = hdul[0].data  
            img = data_cube[0, :, :]  
            
        if bin_centers is None:
            ny, nx = img.shape
            center_y, center_x = ny // 2, nx // 2
            
            y, x = np.indices((ny, nx))
            
            r_matrix = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            r_flat = r_matrix.ravel()
            
            nbins = int(np.min([center_x, center_y]))
            r_bins = np.arange(0, nbins + 1, 1)
            bin_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
            
        img_flat = img.ravel()
        radial_profile = []
        
        for i in range(len(r_bins) - 1):
            mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
            if np.any(mask):
                radial_profile.append(np.mean(img_flat[mask]))
            else:
                radial_profile.append(0.0)
                
        all_profiles.append(radial_profile)
        processed_files += 1

    print(f"\nProcessing complete. Successfully analyzed {processed_files}/{end_num - start_num + 1} files.")
    return bin_centers, all_profiles

if __name__ == "__main__":
    # path to the directory containing the FITS files
    data_folder = "/zfs/helios/filer0/oporth/share/MAD-rhovfix/a+15o16/fits"
    
    # fits file numbers
    start_frame = 1500
    end_frame = 3000
    
    bin_centers, all_profiles = calculate_radial_profiles(start_num=start_frame, end_num=end_frame, base_dir=data_folder)
    
    if all_profiles:
        all_profiles = np.array(all_profiles)
        
        pixel_to_rg = 0.1
        bin_centers_rg = bin_centers * pixel_to_rg
        
        mean_profile = np.mean(all_profiles, axis=0)
        
        std_profile = np.std(all_profiles, axis=0)
        
        plt.figure(figsize=(9, 6))
        
        for i, profile in enumerate(all_profiles):
            plt.plot(bin_centers_rg, profile, color='gray', alpha=0.1, lw=1, zorder=1)
            
        avg_line, = plt.plot(bin_centers_rg, mean_profile, color='darkblue', lw=2.5, zorder=4)
        
        plt.fill_between(bin_centers_rg, mean_profile - std_profile, mean_profile + std_profile, 
                         color='blue', alpha=0.25, zorder=3)

        snapshot_leg = mlines.Line2D([], [], color='gray', alpha=0.6, lw=1.5, label='Individual Snapshots')
        
        sigma_patch = plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.25)

        plt.legend(
            [snapshot_leg, avg_line, sigma_patch], 
            ['Individual Snapshots', 'Time-Averaged Profile', '1-$\sigma$ Temporal Variation'],
            loc='upper right', 
            handlelength=2.5, 
            fontsize=18
        )
        
        plt.xlabel('Radius $r$ ($r_g$)', fontsize=20)
        plt.ylabel('Average Intensity (Jy/pixel)', fontsize=20)
        plt.title(f'Radial Intensity Profile', fontsize=20)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        os.makedirs('output_plots', exist_ok=True)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tight_layout()
        plt.savefig('output_plots/average_intensity_sharp.png', dpi=300, bbox_inches='tight')
        
        # Show the plot
        plt.show()
    else:
        print("Error: No data available to plot. Double check that the folder 'a-15o16/fits' exists in your current working directory and contains the FITS files.")