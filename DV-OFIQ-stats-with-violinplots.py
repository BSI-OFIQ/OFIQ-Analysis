import os
from typing import List, Union
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Define the monochrome color for violinplot, if color args.color is false
ALL_BASE_COLOR = "#CCCCCC"
# MONO_BASE_COLOR = "#CCCCCC"
MONO_BASE_COLOR = "powderblue"
# MONO_BASE_COLOR = "white"
GENDER_ORDER = ["All", "Female", "Male"]
GENDER_COLOR_MAP = {
    "All": ALL_BASE_COLOR,
    "Female": "#F41C1E",   
    "Male": "#56ABCF",
}
AGE_ORDER = ["All", "0-20", "20-30", "30-40", "40-50", "50+"]
AGE_COLOR_MAP = {
    "All": ALL_BASE_COLOR,
    "0-20": "#FFA500",   
    "20-30": "#56ABCF",
    "30-40": "#F41C1E",
    "40-50": "#845EC2",
    "50+":   "#d7bd96",
}
GLASSES_ORDER = ["All", "No-Glasses", "Glasses"]
GLASSES_COLOR_MAP = {
    "All": ALL_BASE_COLOR,
    "No-Glasses": "#FFA500",   
    "Glasses": "#56ABCF",
}
SKINTONE_ORDER = ["All"] + [str(i) for i in range(1, 11)]
SKINTONE_COLOR_MAP = {
    "All": ALL_BASE_COLOR,
    "1":  "#f6ede4",
    "2":  "#f3e7db",
    "3":  "#f7ead0",
    "4":  "#eadaba",
    "5":  "#d7bd96",
    "6":  "#a07e56",
    "7":  "#825c43",
    "8":  "#604134",
    "9":  "#3a312a",
    "10": "#292420",
}

def parse_args():
    #    parser = argparse.ArgumentParser(description="Analyze measure grouped by gender (m/f) and create boxplot.")
    parser = argparse.ArgumentParser(
        description="Analyze measure grouped by demographic variable to create violin-plots and percentiles."
    )
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--variable",
        required=True,
        help="Demographic variable of interest (gender/ethnicity/skintone/age/glasses) ",
    )
    parser.add_argument(
        "--measure", required=True, help="Column name of the measure to analyze"
    )
    parser.add_argument(
        "--color", required=True, help="Violing plots in color"
    )
    parser.add_argument(
        "--output_folder", required=True, help="Folder where the plots will be saved"
    )
    return parser.parse_args()


# Calculates the Gini Coefficinet (GC) Score for a given Quality Score Distribution
def gini_coefficient(x: Union[List[Union[float, int]], np.ndarray]) -> float:
    """
    Note:
    -----
    Calculates the Gini coefficient (GC) for a given input list of descriptive biometric quality scores.
    Descriptive (e.g. mean or median) quality scores are expected as input, with each input value representing
    a specific demographic distribution.
    To calculate the proposed GC-SQFR and GC-CSQFR, which follow a “higher is better” semantic in the range [0,1],
    where a value of 1 represents maximum fairness and a value of 0 represents minimum fairness,
    one needs to calculate 1 - gini_coefficient(x) or (1 - gini_coefficient(x))**3 respecitively.

    Parameters:
    ----------
    x : List[float], List[int] or array-like
        A list or array of numerical values (e.g. mean or median) quality scores for which to calculate the Gini coefficient.
        Must contain more than one element.

    Returns:
    -------
    float
        The Gini coefficient for the input list of quality scores.

    Raises:
    ------
    ValueError
        If the input list contains a single quality score or if the denominator is zero.
    """
    n = len(x)
    if n <= 1:
        raise ValueError(
            "The input list must contain more than one quality score to compute the Gini coefficient."
        )

    # Calculate the numerator as the sum of absolute differences between all pairs (i, j)
    numerator = sum(abs(x[i] - x[j]) for i in range(n) for j in range(n))

    # Calculate the denominator: 2 * (n^2) * mean of x
    denominator = 2 * (n**2) * np.mean(x)

    if denominator == 0:
        raise ZeroDivisionError("Denominator is zero")

    # Compute the raw Gini coefficient
    gc = numerator / denominator

    # Correction factor N/(N-1) to account for smaller group numbers (n)
    correction_factor = n / (n - 1)

    # Return the adapted Gini coefficient
    return gc * correction_factor  # type: ignore


# Calculates the Low-Weighted-Mean (LWM) Score for a given Quality Score Distribution Qdi.
def low_weighted_mean_score(
    Qdi: Union[List[Union[float, int]], np.ndarray],
    QD: Union[List[Union[float, int]], np.ndarray],
) -> float:
    """
    Note:
    -----
    The obtained LWM scores for the set of demographic groups to be evaluated can be used as an alternative input for the Gini coefficient.
    The following is assuming quality scores as integers in [0, 100], i.e. 101 possible values, which means the relevant thresholds are integers in [0, 100), i.e. 100 possible values.
    If your underlying quality score range is different, you need to change the value of the variables QS_UPPER_LIMIT and N_POSSIBLE_SCORES respectively.

    Based on ISO/IEC 29794-1 and the code from https://github.com/dasec/QA-Fairness-Measures

    Parameters:
    ----------
    Qdi : List[float], List[int] or array-like
        A quality score distribution Q for a single demographic group di

    QD : List[float], List[int] or array-like
        A list or array of all quality scores Q across the union set D of demographic groups to be evaluated.
    Returns:
    -------
    float
        The Low-Weighted-Mean (LWM) Score for a single demographic group.
    """
    QS_UPPER_LIMIT = 100  # Adjust if required
    N_POSSIBLE_SCORES = QS_UPPER_LIMIT + 1  # Adjust if required

    # Initialization of the quality score weight embedding
    score_embedding = np.zeros(N_POSSIBLE_SCORES, dtype=np.float64)
    # The overall min and max quality score for normalization
    min_qs, max_qs = min(QD), max(QD)

    # Calculate the weight sums for the quality_score instances in the quality_scores distribution:
    for quality_score in Qdi:
        weight = (
            1 - ((quality_score - min_qs) / (max_qs - min_qs))
        )  # Inverted min-max normalization for the possibility of generalization to quality scores on arbitrary scales
        score_embedding[quality_score] += weight

    # Normalization
    score_embedding_normalized = score_embedding / np.sum(score_embedding)

    return (
        100
        * (np.arange(QS_UPPER_LIMIT + 1) / QS_UPPER_LIMIT)
        @ score_embedding_normalized
    )  # type: ignore


# Calculates the Low-Weighted-Mean-Demographic-Differential (LWM-DD) Score
def lwm_dd_metric(
    df: pd.DataFrame, demog_col: str, metric_col: str, *, return_score: bool = False
):
    """
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe that already contains the column ``metric_col``.
    demog_col : str
        Column used for grouping (e.g. 'age_group', 'gender', 'glasses', …).
    metric_col : str
        Column that holds the quality values (scalar or native).
    return_score : bool, default=False
        If True, the function also returns the Gini coefficient (gc).

    Returns
    -------
    lwm_series : pd.Series
        Series with index = demographic groups + 'All' and name='LWM'.
    lwm_dd : float, optional
        Following the definition in https://www.iso.org/obp/ui/#iso:std:iso-iec:2382:-37:ed-3:v1:en:term:37.09.28
        Demographic Differential "extent of difference in outcome of a biometric system across socially recognized sectors of the population"
        Lower is better semantic
    """
    # Gather the quality values for each group
    group_vals = {}
    for grp, subdf in df.groupby(demog_col, observed=True):
        vals = pd.to_numeric(subdf[metric_col], errors="coerce").dropna().values
        if vals.size > 0:
            group_vals[grp] = vals
    # Union of all values (QD) – needed by the LWM formula
    QD = np.concatenate(list(group_vals.values()))

    # Low‑Weighted‑Mean per group
    lwm_per_group = {g: low_weighted_mean_score(v, QD) for g, v in group_vals.items()}
    # Gini coefficient (optional)
    gc = gini_coefficient(list(lwm_per_group.values()))
    # lwm_dd = 1- gc
    lwm_dd = gc

    # Assemble the Series – prepend the 'All' row
    all_lwm = low_weighted_mean_score(df[metric_col].values, QD)

    lwm_series = pd.Series(lwm_per_group, name="LWM")
    lwm_series = pd.concat([pd.Series({"All": all_lwm}, name="LWM"), lwm_series])

    # Return what the caller asked for
    if return_score:
        return lwm_series, lwm_dd
    return lwm_series


# def write_threshold_table(df: pd.DataFrame, measures: list, var_name: str, out_folder: str) -> None:
def write_threshold_table(
    df: pd.DataFrame, dataset, measures: list, out_folder: str
) -> None:
    """
    Create a CSV file that contains the five operational thresholds
    (0.1 %, 1 %, 5 %, 10 %, 15 %) for *all* quality‑measure columns.

    Parameters
    ----------
    df        : DataFrame – the **original** dataframe (already filtered
                for the current demographic variable, e.g. only rows
                where gender is ‘f’ or ‘m’, only rows where age is valid,
                etc.).
    measures  : list of column names (strings) that hold the quality
                scores (e.g. ['UnifiedQualityScore.scalar',
                'BackgroundUniformity.scalar', …]).
    var_name  : name of the demographic variable that is being processed
                (used only to build a readable file name).
    out_folder: folder where the CSV should be written.
    """
    # 1.  Compute the five percentiles for every measure
    available = [m for m in measures if m in df.columns]
    rows = []
    for m in available:
        arr = np.asarray(pd.to_numeric(df[m], errors="coerce").dropna())
        if arr.size == 0:
            rows.append([m] + ["-"] * 5)
        else:
            rows.append(
                [
                    m,
                    int(np.percentile(arr, 0.1)),
                    int(np.percentile(arr, 1)),
                    int(np.percentile(arr, 5)),
                    int(np.percentile(arr, 10)),
                    int(np.percentile(arr, 15)),
                ]
            )
    thresh_df = pd.DataFrame(
        rows, columns=["Measure", "0,1%", "1%", "5%", "10%", "15%"]
    )

    # 3.  Write the CSV – the file name tells you which DV
    out_path = Path(out_folder) / f"{dataset}-thresholds.csv"
    thresh_df.to_csv(out_path, index=False, sep=";")
    print(f"Threshold table written to: {out_path}")


def create_violinplot(
    df_plot,
    variable,
    measure,
    col_min,
    col_max,
    is_native,
    lwm_dd,
    is_color,
    output_folder,
):
    """
    Create a violin plot for any demographic variable and save it as PNG and SVG.

    Parameters:
    - df_plot (pd.DataFrame): Long-form DataFrame with columns 'group' and 'value'.
    - variable (str): Demographic variable name (e.g. 'gender', 'age', 'skintone', 'glasses').
      Used as the file-name suffix and to look up config from VARIABLE_CONFIG.
    - measure (str): Quality measure column name, used as the plot title and file-name prefix.
    - col_min, col_max: Y-axis limits.
    - is_native (bool): True for native values, False for scalar (0–100).
    - lwm_dd (float): LWM-DD fairness score, appended to the file name when not native.
    - is_color (bool): True → use the variable's colour palette; False → monochrome.
    - output_folder (str): Directory where PNG and SVG files are written.
    """
    VARIABLE_CONFIG = {
        "gender":   {"order": GENDER_ORDER,   "color_map": GENDER_COLOR_MAP,   "xlabel": "Gender"},
        "age":      {"order": AGE_ORDER,       "color_map": AGE_COLOR_MAP,      "xlabel": "Age Group"},
        "glasses":  {"order": GLASSES_ORDER,   "color_map": GLASSES_COLOR_MAP,  "xlabel": "Glasses present"},
        "skintone": {"order": SKINTONE_ORDER,  "color_map": SKINTONE_COLOR_MAP, "xlabel": "Skin Tone Scale"},
    }

    cfg     = VARIABLE_CONFIG.get(variable, {})
    order   = cfg["order"]
    cmap    = cfg["color_map"]
    x_label = cfg.get("xlabel", variable.capitalize())

    palette = cmap if is_color else {g: MONO_BASE_COLOR for g in order}

    figsize    = (6, 4)
    lbl_size   = 15
    ticks_size = 11

    plt.figure(figsize=figsize)
    plt.ylabel("Value",  fontsize=lbl_size)
    plt.xlabel(x_label,  fontsize=lbl_size)
    plt.title(measure)
    plt.ylim(col_min, col_max)
    plt.tick_params(labelsize=ticks_size)

    sns.violinplot(
        data=df_plot,
        x="group",
        y="value",
        hue="group",
        palette=palette,
        order=order,
        hue_order=order,
    )
    plt.tight_layout()

    # Generate path and save
    if is_native:
        path = f"{output_folder}/{measure}_violin.{variable}"
    else:
        path = f"{output_folder}/{measure}_violin.{variable}_lwmdd_{lwm_dd:.3f}"

    for ext in ("png", "svg"):
        fname = f"{path}.{ext}"
        plt.savefig(fname, bbox_inches="tight")
        print(f"Violinplot {ext.upper()} saved to: {fname}")

    plt.close()


def main():
    args = parse_args()

    # List of allowed variables
    allowed_variables = ["gender", "ethnicity", "skintone", "age", "glasses"]
    allowed_color = ["true", "false"]
    
    if args.variable not in allowed_variables:
        print(
            f"Error: Variable '{args.variable}' is not supported. Must be one of: {', '.join(allowed_variables)}."
        )
        return

    try:
        df_full = pd.read_csv(args.input_csv, delimiter=",")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check if color is defined    
    if args.color not in allowed_color:
        print(
            f"Error: Variable '{args.color}' is not supported. Must be one of: {', '.join(allowed_color)}."
        )
        return


    # Check if the column with variable of interest exists
    if args.variable not in df_full.columns:
        print(f"Error: Column '{args.variable}' not found in the CSV file.")
        return

    # Check if the column with measure if interest exists
    if args.measure not in df_full.columns:
        print(
            f"Error: Column '{args.measure}' not found. Available columns: {list(df_full.columns)}"
        )
        return

    # Check if the .native or .sacalar
    measure = args.measure  # e.g. "UnifiedQualityScore.native" or "...scalar"

    # Detect which suffix was supplied
    if args.measure.endswith(".native"):
        base, _ = args.measure.rsplit(".", 1)  # ('native')
        is_native = True
    elif args.measure.endswith(".scalar"):
        base, _ = args.measure.rsplit(".", 1)  # ('scalar')
        is_native = False
    else:
        # The user gave a name without a recognised suffix
        raise ValueError(
            f"The measure '{measure}' must end with '.native' or '.scalar'"
        )

    # Construct the actual column name and verify it exists
    native_col = f"{base}.native"
    scalar_col = f"{base}.scalar"
    if is_native and native_col in df_full.columns:
        metric_col = native_col
    elif not is_native and scalar_col in df_full.columns:
        metric_col = scalar_col
    else:  # fallback – try the other suffix
        raise ValueError(
            f"Neither '{scalar_col}' nor '{native_col}' found in the CSV file."
        )

    # Clean native values (remove the “thousand‑separator” dot)
    if is_native:
        # turn strings like 28.088.387 into a proper float 28.088387
        df_full[metric_col] = (
            df_full[metric_col]
            .astype(str)
            .str.replace(
                r"\.(?=\d{3}\b)", "", regex=True
            )  # drop thousand‑separator dot
            .astype(float)
        )

    # Compute min / max **only** when we are using the native column
    if is_native:
        col_min = df_full[metric_col].min()
        col_max = df_full[metric_col].max()
    else:
        # values are already normalised to 0‑100, the limits are fixed
        col_min = 0
        col_max = 100

    # Color plots
    args = parse_args()
    is_color = args.color.lower() == "true"

    # Identify the dataset that is processed
    base_name = os.path.basename(args.input_csv)
    dataset_name = os.path.splitext(base_name)[0]

    # Build a long‑form DataFrame that the violin‑plot
    # First we keep a copy for the threshold table
    df_full_for_thresholds = df_full.copy()  # <-- this will be used by the helper

    # Second we create a copy that contains only the columns we need.
    # The “group” column will be filled later depending on the
    # selected demographic variable.
    df_plot = df_full[[metric_col]].copy()
    df_plot.rename(columns={metric_col: "value"}, inplace=True)

    # Compute the threshold table for all columns that contain a quality measure
    measure_cols = [
        "UnifiedQualityScore.scalar",
        "BackgroundUniformity.scalar",
        "IlluminationUniformity.scalar",
        "LuminanceMean.scalar",
        "LuminanceVariance.scalar",
        "UnderExposurePrevention.scalar",
        "OverExposurePrevention.scalar",
        "DynamicRange.scalar",
        "Sharpness.scalar",
        "NoCompressionArtifacts.scalar",
        "NaturalColour.scalar",
        "SingleFacePresent.scalar",
        "EyesOpen.scalar",
        "MouthClosed.scalar",
        "EyesVisible.scalar",
        "MouthOcclusionPrevention.scalar",
        "FaceOcclusionPrevention.scalar",
        "InterEyeDistance.scalar",
        "HeadSize.scalar",
        "LeftwardCropOfFace.scalar",
        "RightwardCropOfFace.scalar",
        "MarginAboveTheFace.scalar",
        "MarginBelowTheFace.scalar",
        "HeadPoseYawFrontal.scalar",
        "HeadPosePitchFrontal.scalar",
        "HeadPoseRollFrontal.scalar",
        "ExpressionNeutrality.scalar",
        "NoHeadCoverings.scalar",
    ]
    # Keep only those that are actually in the dataframe
    available_measures = [m for m in measure_cols if m in df_full.columns]
    missing_measures = [m for m in measure_cols if m not in df_full.columns]
    if missing_measures:
        print(
            "The following measures are not present in the input file and will be skipped:"
        )
        print(missing_measures)

    # Report the DV and quality measure of interest
    print("################################################")
    print("ISO/IEC 29794-1 Python script to analyse OFIQ DV")
    print("Input file is: ", dataset_name)
    print("Demographic variable is: ", args.variable)
    print("OFIQ quality measure is: ", args.measure)
    # Feedback on the suffix
    if is_native:
        print("You are working with the *native* quality values (real values).")
    else:
        print(
            "You are working with the *scalar* (mapped) quality component values (integer in the range 0 to 100)."
        )

    # Handle different demographic variables using if-else
    if args.variable == "gender":
        # Process gender: 'm' and 'f'
        df = df_full[["gender", metric_col]].dropna()
        df = df[df["gender"].isin(["m", "f"])]
        labels = ["f", "m"]
        groups = {
            "All": df[metric_col],
            "Female": df[df["gender"] == "f"][metric_col],
            "Male": df[df["gender"] == "m"][metric_col],
        }
        # Temp - dataframe and keep only rows where gender is known and map to readable names
        df_tmp = df[["gender", metric_col]].dropna()
        df_tmp = df_tmp[df_tmp["gender"].isin(["f", "m"])]
        df_tmp["group"] = df_tmp["gender"].map({"f": "Female", "m": "Male"})
        df_plot = df_tmp[["group", metric_col]].rename(columns={metric_col: "value"})

        demog_col = args.variable

        # Statistics for the whole data set ("All")
        overall_stats = df[metric_col].agg(
            count="count", mean="mean", median="median", std="std", min="min", max="max"
        )
        # Turn the Series into a one‑row DataFrame and give it the label “All”
        overall_stats = overall_stats.to_frame().T  # shape (1, 6)
        overall_stats.index = ["All"]  # row name = All

        # Group data
        grouped = df.groupby("gender")[metric_col]
        # result = grouped.agg(count='count', mean='mean', median='median', std='std')
        group_stats = grouped.agg(
            count="count", mean="mean", median="median", std="std", min="min", max="max"
        )

        # Put “All” on top of the age‑group table
        result = pd.concat([overall_stats, group_stats])
        # Optional: keep the original order of the index (All, 0‑20, 20‑30 …)
        result = result.reindex(["All"] + labels)  # ensures the exact order

        # Prepare data for violin plot
        data_all = df[metric_col]
        data_female = df[df["gender"] == "f"][metric_col]
        data_male = df[df["gender"] == "m"][metric_col]

        groups = {"All": data_all, "Female": data_female, "Male": data_male}

        # Create a long-form DataFrame for plotting
        df_plot = pd.concat(
            [
                pd.DataFrame({"group": "All", "value": data_all}),
                pd.DataFrame({"group": "Female", "value": data_female}),
                pd.DataFrame({"group": "Male", "value": data_male}),
            ]
        )

        groups_order = GENDER_ORDER

        # Compute thresholds
        thresholds = {}
        for group, data in groups.items():
            arr = np.asarray(data.dropna())
            if arr.size == 0:
                thresholds[group] = {
                    "0,1%": None,
                    "1%": None,
                    "5%": None,
                    "10%": None,
                    "15%": None,
                }
            else:
                thresholds[group] = {
                    "0,1%": np.percentile(arr, 0.1),
                    "1%": np.percentile(arr, 1),
                    "5%": np.percentile(arr, 5),
                    "10%": np.percentile(arr, 10),
                    "15%": np.percentile(arr, 15),
                }
        # Call function and write thresholds to csv
        write_threshold_table(
            df_full_for_thresholds,
            dataset_name,
            available_measures,
            out_folder=args.output_folder,
        )

        # Fairness measures - Compute LWM-DD - only for the native case
        if not is_native:
            lwm_series, lwm_dd = lwm_dd_metric(
                df, demog_col, metric_col, return_score=True
            )
            lwm_series.index = lwm_series.index.map({"f": "Female", "m": "Male", "All": "All"})
            result = result.reindex(groups_order) 
        else:
            lwm_dd = 0
            print("LWM-DD only computed for scalar.")

    elif args.variable == "skintone":
        # Define bins and labels for skin tone groups (Monk Scale 1–10)
        # Ensure skintone is numeric
        df = df_full[["skintone", metric_col]].dropna()
        df["skintone"] = pd.to_numeric(
            df["skintone"], errors="coerce"
        )  # make sure it is int
        df = df.dropna(subset=["skintone"])  # Remove NaN
        # Temp - dataframe and keep only rows where skintone is present
        df_tmp = df[["skintone", metric_col]].dropna()
        # make sure it is really an integer 1‑10
        df_tmp["group"] = pd.to_numeric(df_tmp["skintone"], errors="coerce").astype(
            "Int64"
        )
        df_plot = df_tmp[["group", metric_col]].rename(columns={metric_col: "value"})

        demog_col = args.variable

        # Create skintone_group from skintone for statistics
        bins = list(range(1, 12))
        labels = [str(i) for i in range(1, 11)]
        df["skintone_group"] = pd.cut(
            df["skintone"], bins=bins, labels=labels, right=False
        )

        # Statistics for the whole data set ("All")
        overall_stats = (
            df_plot["value"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .to_frame()
            .T
        )
        overall_stats.index = ["All"]  # row name = All

        # Group data
        grouped = df.groupby("skintone")[metric_col]
        # result = grouped.agg(count='count', mean='mean', median='median', std='std')
        group_stats = df_plot.groupby("group")["value"].agg(
            ["count", "mean", "median", "std", "min", "max"]
        )

        # Put “All” on top of the age‑group table
        result = pd.concat([overall_stats, group_stats])
        # Optional: keep the original order of the index (All, 0‑20, 20‑30 …)
        order = ["All"] + [str(i) for i in range(1, 11)]
        result = result.reindex(order)

        # Define skin-tone order (Scale 1-10)
        skintone_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ordered_tones = list(range(1, 11))
        df["skintone"] = df["skintone"].astype(int)
        tone_label_map = {t: str(t) for t in ordered_tones}

        # Build the groups for plotting
        groups = {"All": df[metric_col]}
        for skintone_group in skintone_order:
            group_data = df[df["skintone_group"] == skintone_group][metric_col]
            groups[skintone_group] = group_data
        groups = {"All": df[metric_col].copy()}
        for tone in ordered_tones:
            mask = df["skintone"] == tone
            groups[tone_label_map[tone]] = df.loc[mask, metric_col].copy()

        # Create a long-form DataFrame for plotting
        df_plot = pd.concat(
            [
                pd.DataFrame({"group": grp, "value": vals})
                for grp, vals in groups.items()
            ],
            ignore_index=True,
        )
        # Set order for plotting
        groups_order = ["All"] + [tone_label_map[t] for t in ordered_tones]

        # per‑tone groups
        greys = sns.color_palette("PuRd", n_colors=10)  # 10 distinct grey tones
        palette = {"All": "#DDDDDD"}  # neutral colour for “All”
        for t, col in zip(ordered_tones, greys):
            palette[tone_label_map[t]] = col  # type: ignore

        # Compute thresholds
        thresholds = {}
        for group, data in groups.items():
            arr = np.asarray(data.dropna())
            if arr.size == 0:
                thresholds[group] = {
                    "0,1%": None,
                    "1%": None,
                    "5%": None,
                    "10%": None,
                    "15%": None,
                }
            else:
                thresholds[group] = {
                    "0,1%": np.percentile(arr, 0.1),
                    "1%": np.percentile(arr, 1),
                    "5%": np.percentile(arr, 5),
                    "10%": np.percentile(arr, 10),
                    "15%": np.percentile(arr, 15),
                }
        # Call function and write thresholds to csv
        write_threshold_table(
            df_full_for_thresholds,
            dataset_name,
            available_measures,
            out_folder=args.output_folder,
        )

        # Fairness measures - Compute LWM-DD - only for the native case
        if not is_native:
            lwm_series, lwm_dd = lwm_dd_metric(
                df, demog_col, metric_col, return_score=True
            )
            # Index in Strings umwandeln, damit er zu result passt
            lwm_series.index = lwm_series.index.map(str)
            groups_order = ["All"] + [str(t) for t in ordered_tones]
            order = ["All"] + [str(i) for i in range(1, 11)]
            result = result.reindex(order)
        else:
            lwm_dd = 0
            print("LWM-DD only computed for scalar.")

    elif args.variable == "age":
        # Process age: bin into groups
        df = df_full[["age", metric_col]].dropna()

        # Define age bins and labels
        bins = [0, 20, 30, 40, 50, np.inf]
        labels = ["0-20", "20-30", "30-40", "40-50", "50+"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
        # Temp - dataframe and keep only rows where age is known and map to readable names
        df_tmp = df[["age", metric_col]].dropna()
        df_tmp["group"] = pd.cut(df_tmp["age"], bins=bins, labels=labels, right=False)
        df_plot = df_tmp[["group", metric_col]].rename(columns={metric_col: "value"})
        # If the user asked for the “age” demographic, switch to the binned version
        demog_col = args.variable  # original request (e.g. 'age')
        if demog_col == "age":
            demog_col = "age_group"  # <-- use the binned column for grouping

        # Statistics for the whole data set ("All")
        overall_stats = df[metric_col].agg(
            count="count", mean="mean", median="median", std="std", min="min", max="max"
        )
        # Turn the Series into a one‑row DataFrame and give it the label “All”
        overall_stats = overall_stats.to_frame().T  # shape (1, 6)
        overall_stats.index = ["All"]  # row name = All

        # Group data
        # grouped = df.groupby('age_group')[metric_col]
        grouped = df.groupby("age_group", observed=True)[metric_col]
        # result = grouped.agg(count='count', mean='mean', median='median', std='std')
        group_stats = grouped.agg(
            count="count", mean="mean", median="median", std="std", min="min", max="max"
        )

        # Put “All” on top of the age‑group table
        result = pd.concat([overall_stats, group_stats])
        # Optional: keep the original order of the index (All, 0‑20, 20‑30 …)
        result = result.reindex(["All"] + labels)  # ensures the exact order

        # Prepare data for violin plot
        age_order = AGE_ORDER
        groups = {"All": df[metric_col]}
        for age_group in age_order:
            group_data = df[df["age_group"] == age_group][metric_col]
            groups[age_group] = group_data

        # Create a long-form DataFrame for plotting
        df_plot = pd.concat(
            [
                pd.DataFrame({"group": "All", "value": df[metric_col]}),
                pd.DataFrame({"group": df["age_group"], "value": df[metric_col]}),
            ]
        )

        # Set order for plotting
        groups_order = ["All"] + age_order

        # Compute thresholds
        thresholds = {}
        for group, data in groups.items():
            arr = np.asarray(data.dropna())
            if arr.size == 0:
                thresholds[group] = {
                    "0,1%": None,
                    "1%": None,
                    "5%": None,
                    "10%": None,
                    "15%": None,
                }
            else:
                thresholds[group] = {
                    "0,1%": np.percentile(arr, 0.1),
                    "1%": np.percentile(arr, 1),
                    "5%": np.percentile(arr, 5),
                    "10%": np.percentile(arr, 10),
                    "15%": np.percentile(arr, 15),
                }
        # Call function and write thresholds to csv
        write_threshold_table(
            df_full_for_thresholds,
            dataset_name,
            available_measures,
            out_folder=args.output_folder,
        )

        # Fairness measures - Compute LWM-DD - only for the native case
        if not is_native:
            lwm_series, lwm_dd = lwm_dd_metric(
                df, demog_col, metric_col, return_score=True
            )
            lwm_series = lwm_series.reindex(age_order)
        else:
            lwm_dd = 0
            print("LWM-DD only computed for scalar.")

    elif args.variable == "glasses":
        # Process gender: 'm' and 'f'
        df = df_full[["glasses", metric_col]].dropna()
        df = df[df["glasses"].isin(["y", "n"])]

        # Temp - dataframe and keep only rows where glasses is known
        df_tmp = df[["glasses", metric_col]].dropna()
        df_tmp = df_tmp[df_tmp["glasses"].isin(["y", "n"])]
        df_tmp["group"] = df_tmp["glasses"].map({"y": "Glasses", "n": "No-Glasses"})
        df_plot = df_tmp[["group", metric_col]].rename(columns={metric_col: "value"})

        demog_col = args.variable  # original request (e.g. 'age')

        # Group data
        grouped = df.groupby("glasses")[metric_col]
        result = grouped.agg(
            count="count", mean="mean", median="median", std="std", min="min", max="max"
        )

        # Prepare data for violin plot
        data_all = df[metric_col]
        data_glasses = df[df["glasses"] == "y"][metric_col]
        data_noglasses = df[df["glasses"] == "n"][metric_col]

        groups = {
            "All": data_all,
            "No-Glasses": data_noglasses,
            "Glasses": data_glasses,
        }

        # Create a long-form DataFrame for plotting
        df_plot = pd.concat(
            [
                pd.DataFrame({"group": "All", "value": data_all}),
                pd.DataFrame({"group": "No-Glasses", "value": data_noglasses}),
                pd.DataFrame({"group": "Glasses", "value": data_glasses}),
            ]
        )

        groups_order = GLASSES_ORDER

        # Compute thresholds
        # Compute thresholds
        thresholds = {}
        for group, data in groups.items():
            arr = np.asarray(data.dropna())
            if arr.size == 0:
                thresholds[group] = {
                    "0,1%": None,
                    "1%": None,
                    "5%": None,
                    "10%": None,
                    "15%": None,
                }
            else:
                thresholds[group] = {
                    "0,1%": np.percentile(arr, 0.1),
                    "1%": np.percentile(arr, 1),
                    "5%": np.percentile(arr, 5),
                    "10%": np.percentile(arr, 10),
                    "15%": np.percentile(arr, 15),
                }
        # Call function and write thresholds to csv
        write_threshold_table(
            df_full_for_thresholds,
            dataset_name,
            available_measures,
            out_folder=args.output_folder,
        )

        # Fairness measures - Compute LWM-DD
        if not is_native:
            lwm_series, lwm_dd = lwm_dd_metric(
                df, demog_col, metric_col, return_score=True
            )
            lwm_series.index = lwm_series.index.map({"y": "Glasses", "n": "No-Glasses", "All": "All"})
            groups_order = GLASSES_ORDER
            lwm_series = lwm_series.reindex(groups_order)
            result = result.reindex(groups_order)
        else:
            lwm_dd = 0
            print("LWM-DD only computed for scalar.")

    else:
        # Handle other variables (ethnicity, skintone)
        print(f"Warning: Not yet implemented")

    # Print results
    print("\n-----------------------------------------------------------------------")
    print("Statistics for All and by DV-Group:")
    print("-----------------------------------------------------------------------")
    # Overall statistics
    overall = (
        df_plot["value"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .to_frame()
        .T
    )
    # per group statistics
    group_stats = df_plot.groupby("group")["value"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    # combine and order the results
    result = pd.concat([overall, group_stats])
    # Enforce the specific order (e.g. All, 1,2,…,10 for skintone)
    if args.variable == "gender":
        order = ["All"] + ["Female", "Male"]
    elif args.variable == "skintone":
        order = ["All"] + [str(i) for i in range(1, 11)]
    elif args.variable == "age":
        order = ["All"] + ["0-20", "20-30", "30-40", "40-50", "50+"]
    elif args.variable == "glasses":
        order = ["All"] + ["No-Glasses", "Glasses"]

    result = result.reindex(order)
    # Add the LWM
    if not is_native:
        result["LWM"] = lwm_series.reindex(result.index)

    print(result)
    print("-----------------------------------------------------------------------")
    if not is_native:
        print(f"Low-Weighted-Mean-Demographic-Differential (LWM-DD): {lwm_dd:.3f}")
        print("(lower is better)")
        print("-----------------------------------------------------------------------")

    # Create violin plot using the pre-processed df_plot
    create_violinplot(
        df_plot,
        args.variable,
        args.measure,
        col_min,
        col_max,
        is_native,
        lwm_dd,
        is_color,
        args.output_folder,
    )


if __name__ == "__main__":
    main()
