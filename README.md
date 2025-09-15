COMETKiwi Segment Scoring Script
This script performs Quality Estimation (QE) on bilingual translation segments using the COMETKiwi model. It is intended for filtering and scoring translation memory data to support training, fine-tuning, or analysis workflows.
Features
Uses the COMETKiwi model downloaded from Hugging Face.

Filters out:
Empty or short segments
Numeric-only, date-like, and range-like entries
Exact duplicates


Scores each segment using COMETKiwi.
Outputs a scored CSV file.
Saves removed duplicates separately for inspection.

Input Requirements
A CSV file containing bilingual segments.
Source and target columns must be named (e.g., "en-US" and "de-DE").
File should be encoded in UTF-8.

Configuration
Set the following variables in the script:
csv_file     = r"path\to\your\input.csv"
output_path  = r"path\to\your\scored_output.csv"
SRC_COL      = "en-US"   # Source language column
TGT_COL      = "de-DE"   # Target language column
MIN_SRC_WORDS = 2        # Minimum word count for source segments

Usage

Install required packages:
pip install pandas tqdm comet

Run the script:
python score_segments.py

The script will automatically download the COMETKiwi model if not already present.


Output
scored_output.csv: Contains source, target, and COMETKiwi score for each segment.
duplicates.csv: Contains removed duplicate segments.

Notes

The scored file can be used as input for training or fine-tuning translation models.
It is compatible with downstream scripts such as my "visualizer" tool, which segments the scored segments as required and provides graph.
