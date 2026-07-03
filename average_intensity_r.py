import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def calculate_radial_profiles(start_num=2000, end_num=2050, base_dir="a-15o16/fits"):
    """
    Loops through FITS files from start_num to end_num in base_dir, 
    extracts Stokes I, and computes the radial intensity profile for each file.
    """
    all_profiles = []
    bin_centers = None
    processed_files = 0

    print(f"Starting batch processing of files from {start_num} to {end_num}...")
    print(f"Searching in directory: {os.path.abspath(base_dir)}\n")

    for idx in range(start_num, end_num + 1):
        # Dynamically construct the file name based on the sequence format
        filename = f"img_s{idx}_Rh10_i30.fits"
        filepath = os.path.join(base_dir, filename)
        
        # Verify file existence to prevent runtime crashes if a frame is missing
        if not os.path.exists(filepath):
            print(f"  [Skipping] {filename} not found.")
            continue
            
        print(f"  [Processing] {filename}...")
        
        # 1. Load the FITS image data cube
        with fits.open(filepath) as hdul:
            data_cube = hdul[0].data  # Shape: (6, 400, 400)
            img = data_cube[0, :, :]  # Extract Stokes I (Total Intensity)
            
        # 2. Setup coordinate geometry and bins on the first successful file
        if bin_centers is None:
            ny, nx = img.shape
            center_y, center_x = ny // 2, nx // 2
            
            # Create a coordinate grid of (y, x) indices
            y, x = np.indices((ny, nx))
            
            # Calculate the radial distance of each pixel from the center
            r_matrix = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            r_flat = r_matrix.ravel()
            
            # Setup bins up to the closest edge (1 pixel wide increments)
            nbins = int(np.min([center_x, center_y]))
            r_bins = np.arange(0, nbins + 1, 1)
            bin_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
            
        # 3. Calculate the profile for the current frame
        img_flat = img.ravel()
        radial_profile = []
        
        for i in range(len(r_bins) - 1):
            # Create a mask for pixels within the current ring/bin radius
            mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
            if np.any(mask):
                radial_profile.append(np.mean(img_flat[mask]))
            else:
                radial_profile.append(0.0)
                
        all_profiles.append(radial_profile)
        processed_files += 1

    print(f"\nProcessing complete. Successfully analyzed {processed_files}/{end_num - start_num + 1} files.")
    return bin_centers, all_profiles

# --- Execution and Plotting ---
if __name__ == "__main__":
    # Pointing directly to your data folder
    data_folder = "a-15o16/fits"
    
    bin_centers, all_profiles = calculate_radial_profiles(start_num=2000, end_num=2050, base_dir=data_folder)
    
    if all_profiles:
        # Convert list to array for easier calculation
        all_profiles = np.array(all_profiles)
        
        # Compute the overall time-average profile across all frames
        mean_profile = np.mean(all_profiles, axis=0)
        
        # Compute standard deviation to capture variation over time
        std_profile = np.std(all_profiles, axis=0)
        
        plt.figure(figsize=(9, 6))
        
        # Plot individual frame profiles in the background to show variability
        for i, profile in enumerate(all_profiles):
            label = "Individual Snapshots" if i == 0 else ""
            plt.plot(bin_centers, profile, color='gray', alpha=0.15, lw=1, label=label)
            
        # Plot the final averaged radial profile
        plt.plot(bin_centers, mean_profile, color='darkblue', lw=2.5, label='Time-Averaged Profile')
        
        # Fill standard deviation region (shading) to capture fluctuation
        plt.fill_between(bin_centers, mean_profile - std_profile, mean_profile + std_profile, 
                         color='blue', alpha=0.1, label='1-$\sigma$ Temporal Variation')
        
        plt.xlabel('Radius $r$ (pixels)', fontsize=12)
        plt.ylabel('Average Intensity (Jy/pixel)', fontsize=12)
        plt.title('Radial Intensity Profile (Frames 2000 - 2050)', fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')
        # Maakt de getallen op zowel de x- als de y-as groter (grootte 12)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tight_layout()
        
        # Show the plot
        plt.show()
