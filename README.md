# Cyclic Voltammetry Data Analysis

This project is designed to analyze cyclic voltammetry (CV) data from CSV files. The script parses the data, calculates relevant parameters, and generates plots to visualize the results.

## Setup and Usage

### Directory and File Naming

- **`directory_path`**: The path to the directory containing the CSV files.
- **`filename_prefix`**: The prefix of the CSV files to be analyzed.

### Constants and Parameters

- **`TARGET_VOLTAGE_RANGE`**: The target voltage range for analysis.
- **`OFFSET`**: Offset value for voltage.
- **`ELECTRODE_DIAMETER`**: Diameter of the electrode in millimeters.
- **`PLOT_CURRENT_DENSITY`**: Boolean to determine if current density should be plotted instead of current.
- **`PLOT_SIZE`**: Size of the plot.
- **`COLORS`**: Color palette for the plots.
- **`SCAN_RATE_AXES`**: Labels for the scan rate vs peak current plot.
- **`VOLTAMMOGRAM_AXES`**: Labels for the voltammogram plot.
- **`SWEEP_SELECTIONS`**: Segments to be selected for analysis.

### Parsing Functions

- **`parse_cv_file(filename)`**: Parses a single CV file to extract scan rate and peak data.
- **`parse_all_files(directory, prefix)`**: Parses all files in the specified directory with the given prefix.
- **`get_scan_rates_from_files(directory, prefix)`**: Extracts scan rates from all files in the specified directory with the given prefix.

### Data Collection

- **`data_collection`**: A dictionary containing parsed data from all files.
- **`FILE_PATHS`**: List of file paths for the CSV files.
- **`SCAN_RATES`**: List of scan rates extracted from the files.
- **`PLOT_LABELS`**: Labels for the plots based on scan rates.
- **`PEAK_CURRENTS_OX`**: List of oxidation peak currents.
- **`PEAK_CURRENTS_RED`**: List of reduction peak currents.

### DataFrame

A DataFrame is created to store and display the parsed data in a tabular form. The DataFrame includes the following columns:
- **`File`**: The name of the file.
- **`Scan Rate`**: The scan rate extracted from the file.
- **`Oxidation Peak (Ep, Ip)`**: The oxidation peak potential and current.
- **`Reduction Peak (Ep, Ip)`**: The reduction peak potential and current.
- **`Delta Ep`**: The difference between the oxidation and reduction peak potentials.

### Plotting Functions

- **`plot_scan_rate_vs_peak_current(scan_rates, peak_currents_ox, peak_currents_red)`**: Plots the scan rate vs peak current.
- **`plot_voltammogram(file_path, color, label)`**: Plots the voltammogram for a given file.

### Usage

1. Ensure the `directory_path` and `filename_prefix` are correctly set to point to your data files.
2. Run the script to parse the data and generate the plots.
3. Check the printed DataFrame to ensure the identified segments are indeed peaks of the anolyte and not peaks of an impurity.

### Important Note

Be very careful to check that the identified segments are indeed peaks of the anolyte and not the peaks of an impurity. Always verify the printed DataFrame to ensure the accuracy of the parsed data.

