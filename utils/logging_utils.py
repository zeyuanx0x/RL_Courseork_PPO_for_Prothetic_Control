import os
import json
import csv
from typing import Dict, List, Any

def setup_results_dir(env_name: str, base_dir: str = "figures") -> Dict[str, str]:
    """Setup results directory structure
    
    Args:
        env_name: Environment name
        base_dir: Base directory
        
    Returns:
        Dictionary containing subdirectory paths
    """
    # All figures go directly to figures/ directory
    return {
        "main": base_dir,
        "training_curves": base_dir,
        "value_heatmaps": base_dir,
        "effect_sizes": base_dir,
        "combined_figures": base_dir
    }

def generate_filename(prefix: str, env_name: str, suffix: str = ".png", **kwargs) -> str:
    """Generate unified filename
    
    Args:
        prefix: Filename prefix
        env_name: Environment name
        suffix: File suffix
        **kwargs: Additional parameters for filename construction
        
    Returns:
        Generated filename
    """
    # Base filename
    filename = f"{prefix}_{env_name}"
    
    # Add additional parameters to filename
    for key, value in kwargs.items():
        if value is not None:
            filename += f"_{key}_{value}"
    
    # Add suffix
    filename += suffix
    
    return filename

def save_results(results: Dict[str, Any], filename: str, directory: str = ".", format: str = "json"):
    """Save results, supporting multiple formats
    
    Args:
        results: Results dictionary
        filename: Filename (without suffix)
        directory: Save directory
        format: Save format, supporting json, csv, txt
    """
    # Create directory
    os.makedirs(directory, exist_ok=True)
    
    # Build full path
    filepath = os.path.join(directory, f"{filename}.{format}")
    
    # Save according to format
    if format.lower() == "json":
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
    elif format.lower() == "csv":
        # Assume results is a dictionary of lists, where each value is a list
        if results:
            keys = results.keys()
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(keys)
                writer.writerows(zip(*[results[key] for key in keys]))
    elif format.lower() == "txt":
        with open(filepath, "w") as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Results saved as {filepath}")
    return filepath

def update_report(report_path: str, new_content: str, mode: str = "a"):
    """Update report file
    
    Args:
        report_path: Report file path
        new_content: New content
        mode: File open mode
    """
    # Create directory
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Write content
    with open(report_path, mode, encoding="utf-8") as f:
        f.write(new_content)
    
    return report_path