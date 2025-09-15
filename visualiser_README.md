COMETKiwi Score Distribution & Filtering Script
This script analyzes and visualizes the distribution of COMETKiwi scores from a scored CSV file. It also supports filtering segments by score thresholds or custom ranges, and exports the results for downstream use.
Features

Loads a scored CSV file and extracts the cometkiwi_score column.
Computes basic statistics and quantiles.
Plots a histogram of score distribution using matplotlib or seaborn.
Adds vertical lines for quantiles and score cutoffs.
Optionally shades custom score ranges on the histogram.
Exports filtered subsets based on:

Individual score cutoffs (e.g., ≥ 0.80)
Explicit score ranges (e.g., 60–85%)


Adds a score_range column to the main CSV for range-based filtering.

Input Requirements

A CSV file containing a cometkiwi_score column.
UTF-8 encoding is expected.

Configuration
Set the following variables in the script:
csv_path     = r"path\to\your\scored_csv.csv"
score_col    = "cometkiwi_score"
cutoffs      = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
bins         = 40  # or None to use Freedman–Diaconis rule
ranges       = [(60, 85), (85, 100)]  # in percentages or decimals

Usage


Install required packages:
pip install pandas numpy matplotlib seaborn



Run the script:
python visualize_scores.py



The script will generate:

A histogram PNG saved next to the input CSV.
Filtered CSV files for each cutoff and range.
A version of the main CSV with a score_range column.



Output

*_score_hist_YYYYMMDD_HHMMSS.png: Histogram of score distribution.
*_ge_080.csv: Filtered segments with scores ≥ 0.80.
*_range_60-85.csv: Filtered segments within the 60–85% range.
*_with_score_range_YYYYMMDD_HHMMSS.csv: Annotated CSV with score range labels.

Notes

Ranges can be specified in percentages (e.g., 60–85) or decimals (e.g., 0.60–0.85).
Overlapping ranges will be soft-warned in the console.
This script is useful for selecting high-quality segments for training or analysis.
