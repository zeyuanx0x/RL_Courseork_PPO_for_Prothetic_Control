import matplotlib.pyplot as plt
import os
from typing import Dict, List, Any

def setup_plot_style():    
    """Set up unified plotting style (top journal standard)"""
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'lines.linewidth': 2,
        'axes.linewidth': 1,
        'xtick.major.width': 1,
        'ytick.major.width': 1,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black'
    })

def create_figure(figsize: tuple = (15, 12), nrows: int = 1, ncols: int = 1, **kwargs):
    """Create unified figure instance
    
    Args:
        figsize: Figure size
        nrows: Number of rows
        ncols: Number of columns
        **kwargs: Other parameters
        
    Returns:
        fig, axes tuple
    """
    setup_plot_style()
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)

def save_figure(fig: plt.Figure, filename: str, directory: str = ".", bbox_inches: str = 'tight', format: str = 'png'):
    """Save figure, handle path creation and saving logic uniformly
    
    Args:
        fig: Figure instance
        filename: Filename
        directory: Save directory
        bbox_inches: Bounding box setting
        format: File format
    """
    # Create directory
    os.makedirs(directory, exist_ok=True)
    
    # Build complete path
    filepath = os.path.join(directory, filename)
    
    # Save figure
    fig.savefig(filepath, bbox_inches=bbox_inches, dpi=300, format=format)
    plt.close(fig)
    
    print(f"Plot saved as {filepath}")
    return filepath

def plot_returns(returns: List[float], title: str, env_name: str, directory: str = "."):
    """Plot return curve
    
    Args:
        returns: Return list
        title: Figure title
        env_name: Environment name
        directory: Save directory
    """
    setup_plot_style()
    
    plt.figure(figsize=(10, 6))
    plt.plot(returns, color='#1f77b4')
    plt.title(f"{title}: {env_name}")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    filename = f"returns_{env_name}.png"
    return save_figure(plt.gcf(), filename, directory)

def plot_boxplot(data: List[List[float]], labels: List[str], title: str, ylabel: str, env_name: str, directory: str = "."):
    """Plot boxplot
    
    Args:
        data: Data list, each element is a group of data
        labels: Labels for each data group
        title: Figure title
        ylabel: Y-axis label
        env_name: Environment name
        directory: Save directory
    """
    setup_plot_style()
    
    plt.figure(figsize=(10, 6))
    box_plot = plt.boxplot(data, labels=labels, patch_artist=True)
    
    # Set boxplot style
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for patch, color in zip(box_plot['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for whisker in box_plot['whiskers']:
        whisker.set(color='black', linewidth=1.5)
    for cap in box_plot['caps']:
        cap.set(color='black', linewidth=1.5)
    for median in box_plot['medians']:
        median.set(color='black', linewidth=2)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    filename = f"boxplot_{title.lower().replace(' ', '_')}_{env_name}.png"
    return save_figure(plt.gcf(), filename, directory)