import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# ========== Constants and Parameters ==========

# Directory path containing the CSV files
directory_path = r"X:\My Documents\data\CV\exp5\csv\csvred"
# Prefix of the CSV files to be analyzed
filename_prefix = "rhb-exp5"

# Target voltage range for analysis
TARGET_VOLTAGE_RANGE = 0.5
# Offset value for voltage
OFFSET = 0.0
# Diameter of the electrode in millimeters
ELECTRODE_DIAMETER = 3
# Boolean to determine if current density should be plotted instead of current
PLOT_CURRENT_DENSITY = False
# Size of the plot
PLOT_SIZE = (16, 6)
# Color palette for the plots
COLORS = sns.color_palette('tab10')
# Labels for the scan rate vs peak current plot
SCAN_RATE_AXES = ["Scan Rate (V/s)", "Peak Current (mA)" if PLOT_CURRENT_DENSITY else "Peak Current (A)"]
# Labels for the voltammogram plot
VOLTAMMOGRAM_AXES = ["Potential (V vs. Ag/AgCl)",
                     "Current Density (mA/cm^2)" if PLOT_CURRENT_DENSITY else "Current (A)"]
# Segments to be selected for analysis
SWEEP_SELECTIONS = [2, 3]

# Set seaborn style and context for plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})

# ========== Parsing Functions ==========

def parse_cv_file(filename):
    """Parses a single CV file to extract scan rate and peak data."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    scan_rate = None
    # Extract scan rate from the file
    for line in lines:
        if "Scan Rate (V/s)" in line:
            scan_rate = float(line.split('=')[1].strip())
            break

    data = {}
    current_segment = None
    # Extract peak data for segments 2 and 3
    for line in lines:
        if "Segment" in line:
            current_segment = int(line.split()[-1].strip(":"))
            data[current_segment] = []
        elif "Ep =" in line and (current_segment == 2 or current_segment == 3):
            ep = float(line.split('=')[1].replace('V', '').strip())
            ip = float(lines[lines.index(line) + 1].split('=')[1].replace('A', '').strip())
            data[current_segment].append((ep, ip))

    return scan_rate, data.get(2, []), data.get(3, [])

def parse_all_files(directory, prefix):
    """Parses all files in the specified directory with the given prefix."""
    file_pattern = os.path.join(directory, f"{prefix}*.csv")
    files = glob.glob(file_pattern)

    data_collection = {}
    # Parse each file and collect data
    for file in files:
        scan_rate, segment2, segment3 = parse_cv_file(file)
        delta_Ep = None
        if segment2 and segment3:
            delta_Ep = segment3[0][0] - segment2[0][0]
        data_collection[file] = {
            "Scan Rate": scan_rate,
            "Segment 2 Data": segment2,
            "Segment 3 Data": segment3,
            "Delta Ep": delta_Ep
        }

    return data_collection

def get_scan_rates_from_files(directory, prefix):
    """Extracts scan rates from all files in the specified directory with the given prefix."""
    file_pattern = os.path.join(directory, f"{prefix}*.csv")
    files = glob.glob(file_pattern)

    scan_rates = []
    # Extract scan rate from each file
    for file in files:
        scan_rate, _, _ = parse_cv_file(file)
        if scan_rate:
            scan_rates.append(scan_rate)

    return scan_rates

# Parse all files and collect data
data_collection = parse_all_files(directory_path, filename_prefix)

# List of file paths for the CSV files
FILE_PATHS = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if
              file.startswith(filename_prefix) and file.endswith('.csv')]
# List of scan rates extracted from the files
SCAN_RATES = get_scan_rates_from_files(directory_path, filename_prefix)
# Labels for the plots based on scan rates
PLOT_LABELS = [f"Scan Rate: {rate}" for rate in SCAN_RATES]

# List of oxidation peak currents
PEAK_CURRENTS_OX = [entry["Segment 2 Data"][0][1] * -1000 for entry in data_collection.values() if
                    entry["Segment 2 Data"]]
# List of reduction peak currents
PEAK_CURRENTS_RED = [entry["Segment 3 Data"][0][1] * -1000 for entry in data_collection.values() if
                     entry["Segment 3 Data"]]

# Create a DataFrame to store and display information in tabular form
df = pd.DataFrame(columns=["File", "Scan Rate", "Oxidation Peak (Ep, Ip)", "Reduction Peak (Ep, Ip)", "Delta Ep"])

# Populate the DataFrame with parsed data
for idx, (file, data) in enumerate(data_collection.items()):
    oxidation_peak = data["Segment 2 Data"][0] if data["Segment 2 Data"] else None
    reduction_peak = data["Segment 3 Data"][0] if data["Segment 3 Data"] else None
    df.loc[idx] = [file, data["Scan Rate"], oxidation_peak, reduction_peak, data["Delta Ep"]]
    # Extract just the filename from the full path
    df['File'] = df['File'].apply(lambda x: os.path.basename(x))

    # Set the display options to make the output neater
    pd.set_option('display.max_columns', None)  # show all columns
    pd.set_option('display.expand_frame_repr', False)  # prevent wrapping to the next line

# Display the table
print(df)

# ========== Plotting Functions ==========

def plot_scan_rate_vs_peak_current(scan_rates, peak_currents_ox, peak_currents_red):
    """Plots the scan rate vs peak current."""
    # Constants for calculation
    n = 1
    diameter = 0.3
    A = np.pi * (diameter / 2) ** 2
    C = 3e-6
    constant = 2.69e5 * (n ** 1.5) * A * C

    sqrt_scan_rates = np.sqrt(scan_rates)

    # Fit a line to the oxidation peak currents
    z_ox, cov_ox = np.polyfit(sqrt_scan_rates, peak_currents_ox, 1, cov=True)
    p_ox = np.poly1d(z_ox)
    R2_ox = 1 - (np.sum((peak_currents_ox - p_ox(sqrt_scan_rates)) ** 2) / (
                (len(peak_currents_ox)) * np.var(peak_currents_ox, ddof=1)))

    # Fit a line to the reduction peak currents
    z_red, cov_red = np.polyfit(sqrt_scan_rates, peak_currents_red, 1, cov=True)
    p_red = np.poly1d(z_red)
    R2_red = 1 - (np.sum((peak_currents_red - p_red(sqrt_scan_rates)) ** 2) / (
                (len(peak_currents_red)) * np.var(peak_currents_red, ddof=1)))

    # Calculate diffusion coefficients
    D_ox = (z_ox[0] * 1000 / (constant)) ** 2
    D_red = (z_red[0] * 1000 / (constant)) ** 2

    # Annotate the plot with diffusion coefficients
    plt.annotate(f"D_ox = {D_ox:.2e} cm$^{2}$/s", xy=(0.1, 0.9), xycoords="axes fraction")
    plt.annotate(f"D_red = {D_red:.2e} cm$^{2}$/s", xy=(0.1, 0.85), xycoords="axes fraction")

    # Plot peak currents vs square root of scan rates
    if PLOT_CURRENT_DENSITY:
        peak_currents_ox_density = [i / A for i in peak_currents_ox]
        peak_currents_red_density = [i / A for i in peak_currents_red]
        plt.scatter(sqrt_scan_rates, peak_currents_ox_density, label='Oxidation Peaks', color='blue')
        plt.plot(sqrt_scan_rates, p_ox(sqrt_scan_rates) / A, "b--", label=f"Oxidation Fit (R$^{2}$={R2_ox:.4f})")
        plt.scatter(sqrt_scan_rates, peak_currents_red_density, label='Reduction Peaks', color='red')
        plt.plot(sqrt_scan_rates, p_red(sqrt_scan_rates) / A, "r--", label=f"Reduction Fit ($^{2}$={R2_red:.4f})")
        plt.ylabel("Peak Current Density (mA/cm$^2$)")
    else:
        plt.scatter(sqrt_scan_rates, peak_currents_ox, label='Oxidation Peaks', color='blue')
        plt.plot(sqrt_scan_rates, p_ox(sqrt_scan_rates), "b--", label=f"Oxidation Fit (R$^{2}$={R2_ox:.4f})")
        plt.scatter(sqrt_scan_rates, peak_currents_red, label='Reduction Peaks', color='red')
        plt.plot(sqrt_scan_rates, p_red(sqrt_scan_rates), "r--", label=f"Reduction Fit (R$^{2}$={R2_red:.4f})")
        plt.ylabel("Peak Current (mA)")

    plt.xlabel("Square Root of Scan Rate (V/s)$^{0.5}$")
    plt.legend()
    plt.title('Square Root of Scan Rate vs Peak Current')
    plt.grid(True)
    plt.show()

def plot_voltammogram(file_path, color, label):
    """Plots the voltammogram for a given file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the start line of the data
    start_line = lines.index('Potential/V, Current/A\n')
    data = pd.read_csv(file_path, skiprows=start_line, delimiter=',', na_values=["", " "])
    data.columns = [col.strip() for col in data.columns]

    # Convert current to current density if electrode diameter is provided
    if ELECTRODE_DIAMETER:
        radius_cm = ELECTRODE_DIAMETER / 20  # Convert mm to cm for radius
        electrode_surface_area = np.pi * (radius_cm ** 2)  # Area in cm^2
        data['Current/A'] /= electrode_surface_area
        data['Current/A'] *= 1000  # Convert from A/cm^2 to mA/cm^2
        y_label = "Current Density (mA cm$^{-2}$)"  # Updated y-label with LaTeX formatting
    else:
        y_label = "Current (A)"

    # Multiply the current values by -1
    data['Current/A'] *= -1

    # Plot the voltammogram
    plt.plot(data['Potential/V'], data['Current/A'], color=color, label=label)
    plt.xlabel("Potential (V vs. Ag/AgCl)")
    plt.ylabel(y_label)
    plt.title("Cyclic Voltammetry")
    plt.legend()

# Use the function:
data_collection = parse_all_files(directory=directory_path, prefix=filename_prefix)

# Generate side by side plots
fig, ax = plt.subplots(1, 2, figsize=PLOT_SIZE)
# Left plot (Cyclic Voltammograms)
plt.sca(ax[0])
for file_path, color, label in zip(FILE_PATHS, COLORS, PLOT_LABELS):
    plot_voltammogram(file_path, color, label)

# Right plot (Scan Rate vs Peak Current)
plt.sca(ax[1])
plot_scan_rate_vs_peak_current(SCAN_RATES, PEAK_CURRENTS_OX, PEAK_CURRENTS_RED)

plt.tight_layout()

# Save the figure before displaying it
plt.savefig('negative_D.png', dpi=300, bbox_inches='tight')

plt.show()