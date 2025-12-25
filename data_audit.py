import pandas as pd
import numpy as np
import argparse

# Define minimum data requirements for each analysis type
# Updated requirements based on unified CSV Schema design
REQUIREMENTS = {
    "statistical_summary": ["Metric", "Algorithm", "Mean", "Lower_95CI", "Upper_95CI"],
    "perturbation": ["perturb_type", "perturb_level", "episode_return"],
    "safety": ["termination_reason", "episode_return"],
    "contribution": ["mode", "episode_return"],
    "domain": ["domain_condition", "original_domain", "episode_return"],
    "multitask": ["task_id", "episode_return"],
    "hyperparam": ["hp_name", "hp_value", "episode_return"],
}

# Define analysis type descriptions
description = {
    "statistical_summary": "Statistical Summary Analysis (Mean ± 95% CI)",
    "perturbation": "Perturbation Robustness Analysis",
    "safety": "Safety/Failure Mode Analysis",
    "contribution": "Contribution/Human-Machine Collaboration Analysis",
    "domain": "Domain Randomization and Generalization Analysis",
    "multitask": "Multitask Evaluation",
    "hyperparam": "Hyperparameter Sensitivity Analysis",
}

# Define data validity check function
def check_data_validity(df, requirements):
    """Check if data meets analysis requirements"""
    # 1. Column existence check
    missing_cols = [col for col in requirements if col not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # 2. Data validity check
    # Check if each column has ≥2 distinct values (non-NaN)
    for col in requirements:
        # Number of unique values after excluding NaN
        non_nan_vals = df[col].dropna()
        unique_vals = non_nan_vals.nunique()
        if unique_vals < 2:
            return False, f"Column {col} has <2 unique values ({unique_vals})"
    
    # Check if each column is not all NaN/all 0
    for col in requirements:
        if df[col].isna().all():
            return False, f"Column {col} is all NaN"
        if pd.api.types.is_numeric_dtype(df[col]):
            non_nan_vals = df[col].dropna()
            if len(non_nan_vals) > 0 and (non_nan_vals == 0).all():
                return False, f"Column {col} is all 0 (excluding NaN)"
    
    # 3. Sample quantity check (non-NaN samples)
    # Calculate valid samples: at least non-NaN values in all required columns
    valid_samples = df.dropna(subset=requirements)
    if len(valid_samples) < 5:
        return False, f"Insufficient valid samples ({len(valid_samples)} < 5)"
    
    return True, "OK"

# Define analysis readiness report generation function
def generate_readiness_report(df):
    """Generate analysis readiness report"""
    print("=== Analysis Readiness Report ===\n")
    
    readiness = {}
    
    for analysis_type, requirements in REQUIREMENTS.items():
        is_valid, reason = check_data_validity(df, requirements)
        readiness[analysis_type] = {
            "is_ready": is_valid,
            "reason": reason,
            "description": description[analysis_type]
        }
    
    # Output readiness report
    print("| Analysis Type | Status | Description | Reason |")
    print("|---------------|--------|-------------|--------|")
    for analysis_type, info in readiness.items():
        status = "✅ READY" if info["is_ready"] else "❌ NOT READY"
        print(f"| {analysis_type} | {status} | {info['description']} | {info['reason']} |")
    
    print("\n=== Readiness Statistics ===")
    ready_count = sum(1 for info in readiness.values() if info["is_ready"])
    total_count = len(readiness)
    print(f"Ready analyses: {ready_count}/{total_count}")
    print(f"Readiness rate: {(ready_count/total_count)*100:.1f}%")
    
    return readiness

# Main function
def main():
    """Perform data audit"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Data Audit for RL Experiment Results")
    parser.add_argument("--csv", type=str, default="results/all_results.csv", help="Path to results CSV file")
    parser.add_argument("--step-csv", type=str, default="results/step_data.csv", help="Path to step data CSV file")
    parser.add_argument("--report-only", action="store_true", help="Only generate readiness report")
    args = parser.parse_args()
    
    print("=== Data Audit for MyoSuite PPO Results ===\n")
    
    # Read main data file
    csv_path = args.csv
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully loaded {csv_path}")
        print(f"✓ Total rows: {len(df)}")
        print(f"✓ Total columns: {len(df.columns)}")
        print(f"✓ Columns: {', '.join(df.columns)}\n")
    except Exception as e:
        print(f"✗ Failed to load {csv_path}: {e}")
        return
    
    # Check step data file
    step_csv_path = args.step_csv
    step_data_available = False
    if args.report_only:
        print(f"\n=== Step Data Availability Check ===")
        try:
            step_df = pd.read_csv(step_csv_path)
            print(f"✓ Step data available: {step_csv_path}")
            print(f"✓ Step data rows: {len(step_df)}")
            print(f"✓ Step data columns: {len(step_df.columns)}")
            step_data_available = True
        except Exception as e:
            print(f"✗ Step data not available: {e}")
    
    # Generate analysis readiness report
    readiness = generate_readiness_report(df)
    
    # Output final permission table
    print("\n=== Final Permission Table ===\n")
    permission_table = {}
    
    for analysis_type, info in readiness.items():
        if info["is_ready"]:
            permission_table[analysis_type] = "OK"
        else:
            permission_table[analysis_type] = f"SKIP ({info['reason']})"
    
    for analysis_type, status in permission_table.items():
        print(f"{analysis_type}: {status}")
    
    # Check step data dependencies
    if step_data_available:
        print("\n=== Step Data Dependencies ===")
        print("✓ Step data available for safety analysis")
    else:
        print("\n=== Step Data Dependencies ===")
        print("✗ Step data required for detailed safety analysis")
    
    print("\n=== Audit Complete ===")
    print("Only analyses marked as READY can be performed.")
    print("All analyses are checked for: column existence, data validity, and sample sufficiency.")

if __name__ == "__main__":
    main()