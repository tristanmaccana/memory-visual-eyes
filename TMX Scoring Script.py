import re
import pandas as pd
from comet import download_model, load_from_checkpoint
from tqdm import tqdm

# ----------------------------
# Settings
# ----------------------------
# Path to your CSV file
csv_file = r"C:\Users\jmaccat\Downloads\JohnsonControls_JCI TECHCOMMS-en_US-es_ES-2025-09-15\all-techcomms_combined_es.csv"

# Output path
output_path = r"C:\Users\jmaccat\OneDrive - Johnson Controls\Translation\TMX Scoring Script\scored_csv.csv"

# Columns (source and target)
SRC_COL = "en-US"
TGT_COL = "es-ES"

# Minimum word count for source
MIN_SRC_WORDS = 2

# ----------------------------
# Load COMETKiwi model
# ----------------------------
print("⬇️ Loading COMETKiwi model…")
model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)

# ----------------------------
# Read CSV
# ----------------------------
df = pd.read_csv(csv_file, encoding="utf-8")
print("📄 Original DataFrame loaded.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ----------------------------
# Basic cleaning: drop rows with missing/empty source or target
# ----------------------------
before = len(df)
df = df.dropna(subset=[SRC_COL, TGT_COL])

# Ensure strings, strip whitespace, and remove empties
df[SRC_COL] = df[SRC_COL].astype(str).str.strip()
df[TGT_COL] = df[TGT_COL].astype(str).str.strip()
df = df[(df[SRC_COL] != "") & (df[TGT_COL] != "")]
after_basic = len(df)
print(f"🧹 Removed {before - after_basic} rows with missing/empty source/target. Remaining: {after_basic}")

# ----------------------------
# Define vectorized filters
# ----------------------------
# 1) Numeric-only (digits present, no letters)
def numeric_only_series(s: pd.Series) -> pd.Series:
    # Letters in Unicode: [^\W\d_]  (word chars excluding digits and underscore)
    has_letters = s.str.contains(r'[^\W\d_]', regex=True)
    has_digits  = s.str.contains(r'\d', regex=True)
    return has_digits & ~has_letters

# 2) Date-like (string is ONLY a date)
#    Common formats: 12/05/2021, 2021-05-12, 12.5.21, 5 Jan 2021, January 5, 2021, 05 Aug 25, Aug 2025
month = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)'
ord_suffix = r'(?:st|nd|rd|th)?'

date_patterns = [
    r'\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}',      # 12/05/2021, 12-05-21, 12.05.2021
    r'\d{4}[\/\.\-]\d{1,2}[\/\.\-]\d{1,2}',        # 2021/05/12
    rf'{month}\s+\d{{1,2}}{ord_suffix}(?:,)?\s+\d{{2,4}}',   # January 5, 2021 ; Aug 5 25
    rf'\d{{1,2}}{ord_suffix}\s+{month}(?:,)?\s+\d{{2,4}}',   # 5 Jan 2021 ; 5th January, 2021
    rf'{month}\s+\d{{4}}',                                  # August 2025
    r'\d{4}'                                                # bare year (already caught by numeric-only, but included here)
]
date_like_regex = re.compile(rf'^\s*(?:{"|".join(date_patterns)})\s*$', re.IGNORECASE)

def date_like_series(s: pd.Series) -> pd.Series:
    return s.str.match(date_like_regex, na=False)

# 3) Numeric range-like (string is ONLY a numeric range)
#    Examples: "3-4", "3–4", "10 to 20", "between 5 and 7", "from 5 to 10", "10–12%", "0–40 °C", "12-24V"
num = r'~?\d+(?:[.,]\d+)?'                 # number w/ optional decimal and ~
unit = r'(?:\s?(?:%|°[CF]?|[a-z]{1,3}))?'  # very short unit or %/°C/°F
sep  = r'(?:-|–|—|to|through|thru|±|~)'    # separators indicating a range

range_patterns = [
    rf'{num}{unit}\s*{sep}\s*{num}{unit}',                        # 10–12%, 3-4, 12-24V, 0–40 °C
    rf'from\s+{num}{unit}\s+(?:to|through|thru)\s+{num}{unit}',   # from 5 to 10
    rf'between\s+{num}{unit}\s+(?:and|&)\s+{num}{unit}'           # between 5 and 7
]
range_like_regex = re.compile(rf'^\s*(?:{"|".join(range_patterns)})\s*$', re.IGNORECASE)

def range_like_series(s: pd.Series) -> pd.Series:
    return s.str.match(range_like_regex, na=False)

# ----------------------------
# Apply filters: remove rows where EITHER source or target is
# numeric-only OR date-like OR numeric-range-like
# ----------------------------
src = df[SRC_COL]
tgt = df[TGT_COL]

mask_src_numeric_only = numeric_only_series(src)
mask_tgt_numeric_only = numeric_only_series(tgt)

mask_src_date_like = date_like_series(src)
mask_tgt_date_like = date_like_series(tgt)

mask_src_range_like = range_like_series(src)
mask_tgt_range_like = range_like_series(tgt)

mask_numeric_only = mask_src_numeric_only | mask_tgt_numeric_only
mask_date_like    = mask_src_date_like    | mask_tgt_date_like
mask_range_like   = mask_src_range_like   | mask_tgt_range_like
mask_any = mask_numeric_only | mask_date_like | mask_range_like

# Report breakdown (counts may overlap; union shown separately)
count_numeric_only = int(mask_numeric_only.sum())
count_date_like    = int(mask_date_like.sum())
count_range_like   = int(mask_range_like.sum())
count_union        = int(mask_any.sum())

print(f"🔢 Numeric-only rows to remove: {count_numeric_only}")
print(f"📅 Date-like rows to remove:    {count_date_like}")
print(f"➡️  Range-like rows to remove:   {count_range_like}")
print(f"🧮 Total unique rows removed:    {count_union}")

df = df[~mask_any]
print(f"✅ Remaining after numeric/date/range filters: {len(df)}")

# ----------------------------
# Filter: Remove rows where source has fewer than MIN_SRC_WORDS words
# ----------------------------
df["__src_word_count"] = df[SRC_COL].str.count(r'\b\w+\b')
removed_short = int((df["__src_word_count"] < MIN_SRC_WORDS).sum())
df = df[df["__src_word_count"] >= MIN_SRC_WORDS].drop(columns=["__src_word_count"])
print(f"✂️ Removed {removed_short} rows with < {MIN_SRC_WORDS} words in source. Remaining: {len(df)}")

# ----------------------------
# Remove exact duplicate (source, target) rows, keeping the first
# ----------------------------
# Identify duplicates
duplicate_mask = df.duplicated(subset=[SRC_COL, TGT_COL], keep='first')
duplicates_df = df[duplicate_mask]

# Save duplicates to a file
duplicates_df.to_csv("duplicates.csv", index=False, encoding="utf-8-sig")

# Drop duplicates from the original DataFrame
before_dupes = len(df)
df = df.drop_duplicates(subset=[SRC_COL, TGT_COL], keep='first')
removed_dupes = before_dupes - len(df)

print(f"📎 Removed {removed_dupes} exact duplicate (source, target) rows. Remaining: {len(df)}")
print("✅ Duplicates saved to 'duplicates.csv' for review.")

# ----------------------------
# Prepare segments for scoring
# ----------------------------
segments = (
    df[[SRC_COL, TGT_COL]]
    .rename(columns={SRC_COL: "src", TGT_COL: "mt"})
    .to_dict(orient="records")
)

print("🧪 Sample segments for scoring:")
for i, seg in enumerate(segments[:5]):
    print(f"{i+1}: src='{seg['src']}', mt='{seg['mt']}'")

# ----------------------------
# Score segments with COMETKiwi
# ----------------------------
print("⚙️ Scoring with COMETKiwi…")
raw = model.predict(segments, batch_size=8, gpus=0)  # set gpus=1 if you have a GPU

# ----------------------------
# Robustly normalize/unwrap COMET output
# ----------------------------
from collections.abc import Iterable

def _flatten_list_of_lists(x):
    """Flatten [[...], [...]] -> [...] when all inner elements are floats/ints."""
    if isinstance(x, list) and x and all(isinstance(i, list) for i in x):
        if all(i and all(isinstance(v, (float, int)) for v in i) for i in x):
            return [v for i in x for v in i]
    return x

def extract_seg_scores(raw_obj, expected_n):
    """
    Return (seg_scores_list or None, system_score or None, debug_string).
    Handles dict/tuple/list/batched lists/objects with `.scores`.
    """
    sys_score = None
    dbg = f"type={type(raw_obj)}; "
    seg_scores = None

    try:
        # 1) Dict-like
        if isinstance(raw_obj, dict):
            # common keys observed in COMET versions
            if "scores" in raw_obj and isinstance(raw_obj["scores"], list):
                seg_scores = raw_obj["scores"]
            elif "segments_scores" in raw_obj and isinstance(raw_obj["segments_scores"], list):
                seg_scores = raw_obj["segments_scores"]
            elif "predictions" in raw_obj and isinstance(raw_obj["predictions"], list):
                # some interfaces return a list of per-segment dicts
                preds = raw_obj["predictions"]
                if preds and isinstance(preds[0], dict) and "score" in preds[0]:
                    seg_scores = [p["score"] for p in preds]
            sys_score = raw_obj.get("system_score", None)
            dbg += f"dict; seg_len={None if seg_scores is None else len(seg_scores)}; sys={sys_score}"

        # 2) Tuple: ([scores...], system_score) or similar
        elif isinstance(raw_obj, tuple) and len(raw_obj) == 2:
            # first element could be scores or a dict containing them
            first, second = raw_obj
            if isinstance(first, (list, tuple)) and first and isinstance(first[0], (float, int)):
                seg_scores = list(first)
                sys_score = second if isinstance(second, (float, int)) else None
            elif isinstance(first, dict):
                # reuse dict handler
                tmp_scores, tmp_sys, _ = extract_seg_scores(first, expected_n)
                seg_scores = tmp_scores
                sys_score = tmp_sys if tmp_sys is not None else (second if isinstance(second, (float, int)) else None)
            else:
                # sometimes PL returns ([ [...], [...] ], sys)
                first = _flatten_list_of_lists(first) if isinstance(first, list) else first
                if isinstance(first, list) and first and isinstance(first[0], (float, int)):
                    seg_scores = first
                    sys_score = second if isinstance(second, (float, int)) else None
            dbg += f"tuple; seg_len={None if seg_scores is None else len(seg_scores)}; sys={sys_score}"

        # 3) List: could be list-of-floats or list-of-lists (batches) or list-of-dicts
        elif isinstance(raw_obj, list):
            obj = _flatten_list_of_lists(raw_obj)
            # list-of-dicts with 'score'
            if obj and isinstance(obj[0], dict) and "score" in obj[0]:
                seg_scores = [d["score"] for d in obj]
            # list-of-floats
            elif obj and isinstance(obj[0], (float, int)):
                seg_scores = list(obj)
            dbg += f"list; seg_len={None if seg_scores is None else len(seg_scores)}"

        # 4) Objects with `.scores`
        elif hasattr(raw_obj, "scores"):
            maybe = getattr(raw_obj, "scores")
            if isinstance(maybe, list) and maybe and isinstance(maybe[0], (float, int)):
                seg_scores = maybe
            if hasattr(raw_obj, "system_score") and isinstance(raw_obj.system_score, (float, int)):
                sys_score = raw_obj.system_score
            dbg += f"obj-with-attr; seg_len={None if seg_scores is None else len(seg_scores)}; sys={sys_score}"

        # Final sanity: if seg_scores still None but raw_obj feels like a single score replicated
        if seg_scores is not None and len(seg_scores) != expected_n:
            dbg += f" | ⚠️ length-mismatch seg_len={len(seg_scores)} vs expected={expected_n}"

    except Exception as e:
        dbg += f" | error={e}"

    return seg_scores, sys_score, dbg

seg_scores, system_score, shape_dbg = extract_seg_scores(raw, expected_n=len(df))
print("📊 Scoring complete.")
print("🔎 Raw shape debug:", shape_dbg)
if system_score is not None:
    print("ℹ️ System score:", system_score)

# ----------------------------
# Keep raw output + assign scalar scores
# ----------------------------
df["cometkiwi_raw_output"] = [raw] * len(df)

if seg_scores is not None and len(seg_scores) == len(df):
    df["cometkiwi_score"] = seg_scores
    print(f"✅ Assigned cometkiwi_score for {len(seg_scores)} rows.")
else:
    print("❌ Could not assign cometkiwi_score due to unrecognized structure or length mismatch.")
    print(f"   Parsed seg_scores length: {0 if seg_scores is None else len(seg_scores)} | rows: {len(df)}")
    print("   Keeping 'cometkiwi_raw_output' for inspection.")
    

df = df.drop(columns=["cometkiwi_raw_output"], errors="ignore")

# ----------------------------
# Save
# ----------------------------
print("🧾 Preview of DataFrame before saving:")
print(df.head())
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"💾 File saved successfully at: {output_path}")

