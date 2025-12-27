import matplotlib.pyplot as plt
import numpy as np
import os

# Nature journal style configuration
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'lines.linewidth': 1,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.edgecolor': 'black',
    'legend.handlelength': 1.5,
    'legend.borderaxespad': 0.2,
    'legend.borderpad': 0.2,
    'legend.columnspacing': 1.0,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'text.usetex': False
})

class NatureStylePlotter:
    """Generate figures in Nature journal style"""
    
    def __init__(self, env_name):
        self.env_name = env_name
        self.figures_dir = "figures"
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def plot_return_distribution(self, all_results_df):
        """Plot return distribution boxplot"""
        plt.figure(figsize=(3.5, 2.5))
        
        # Get algorithm list
        algorithms = all_results_df["algo"].unique()
        data = [all_results_df[all_results_df["algo"] == algo]["episode_return"].values for algo in algorithms]
        
        # Plot boxplot
        box_plot = plt.boxplot(data, tick_labels=algorithms, patch_artist=True, widths=0.6)
        
        # Set colors
        colors = ['#1f77b4', '#ff7f0e']
        for patch, color in zip(box_plot['boxes'], colors[:len(algorithms)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for whisker in box_plot['whiskers']:
            whisker.set(color='black', linewidth=0.5)
        for cap in box_plot['caps']:
            cap.set(color='black', linewidth=0.5)
        for median in box_plot['medians']:
            median.set(color='black', linewidth=0.75)
        for flier in box_plot['fliers']:
            flier.set(marker='o', markersize=2, color='red', alpha=0.5)
        
        plt.xlabel('Algorithm')
        plt.ylabel('Episode Return')
        plt.title('Return Distribution')
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"return_distribution_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    def plot_success_rate(self, all_results_df):
        """Plot success rate bar chart"""
        # Assume success rate is the ratio of returns greater than 0
        plt.figure(figsize=(3.5, 2.5))
        
        algorithms = all_results_df["algo"].unique()
        success_rates = []
        for algo in algorithms:
            algo_df = all_results_df[all_results_df["algo"] == algo]
            success_rate = len(algo_df[algo_df["episode_return"] > 0]) / len(algo_df)
            success_rates.append(success_rate)
        
        bars = plt.bar(algorithms, success_rates, color=['#1f77b4', '#ff7f0e'], alpha=0.7, width=0.6)
        
        # Add value labels
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                    f'{rate:.3f}', ha='center', va='bottom', fontsize=7)
        
        plt.xlabel('Algorithm')
        plt.ylabel('Success Rate')
        plt.title('Task Success Rate')
        plt.ylim(0, 1.05)
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"success_rate_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    def plot_seed_comparison(self, all_results_df):
        """Plot performance comparison across different seeds"""
        plt.figure(figsize=(3.5, 2.5))
        
        # Only compare PPO algorithm
        ppo_df = all_results_df[all_results_df["algo"] == "PPO"]
        seeds = ppo_df["seed"].unique()
        
        # Calculate average returns by seed
        seed_means = []
        seed_stds = []
        for seed in seeds:
            seed_df = ppo_df[ppo_df["seed"] == seed]
            seed_means.append(seed_df["episode_return"].mean())
            seed_stds.append(seed_df["episode_return"].std())
        
        x = np.arange(len(seeds))
        plt.errorbar(x, seed_means, yerr=seed_stds, fmt='o', capsize=2, color='#1f77b4', 
                   markerfacecolor='white', markeredgewidth=0.5, linewidth=0)
        
        # Draw mean line
        overall_mean = np.mean(seed_means)
        plt.axhline(y=overall_mean, color='gray', linestyle='--', linewidth=0.5)
        plt.text(len(seeds)-0.5, overall_mean+0.5, f'Mean: {overall_mean:.1f}', 
                ha='right', va='bottom', fontsize=7, color='gray')
        
        plt.xlabel('Seed')
        plt.ylabel('Mean Episode Return')
        plt.title('PPO Performance Across Seeds')
        plt.xticks(x, seeds)
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"seed_comparison_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    def plot_episode_length_vs_return(self, all_results_df):
        """Plot episode length vs return scatter plot"""
        plt.figure(figsize=(3.5, 2.5))
        
        algorithms = all_results_df["algo"].unique()
        colors = ['#1f77b4', '#ff7f0e']
        
        for algo, color in zip(algorithms, colors[:len(algorithms)]):
            algo_df = all_results_df[all_results_df["algo"] == algo]
            plt.scatter(algo_df["episode_len"], algo_df["episode_return"], 
                      s=5, alpha=0.7, color=color, label=algo)
        
        plt.xlabel('Episode Length')
        plt.ylabel('Episode Return')
        plt.title('Episode Length vs Return')
        plt.legend(markerscale=2)
        plt.grid(True)
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"episode_length_vs_return_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    def plot_learning_curve_comparison(self, ppo_stats, sac_stats=None):
        """Plot learning curve comparison"""
        plt.figure(figsize=(3.5, 2.5))
        
        # Plot PPO learning curve
        plt.plot(ppo_stats["returns"], label="PPO", color='#1f77b4', linewidth=0.8)
        
        # Plot SAC learning curve (if available)
        if sac_stats:
            plt.plot(sac_stats["returns"], label="SAC", color='#ff7f0e', linewidth=0.8)
        
        plt.xlabel('Episode')
        plt.ylabel('Episode Return')
        plt.title('Learning Curve Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"learning_curve_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    # Physiology audit related figures
    def plot_physio_force_proxy(self, df):
        """Plot force proxy chart"""
        if "action_force_proxy" not in df.columns:
            return
        
        plt.figure(figsize=(3.5, 2.5))
        plt.plot(df["action_force_proxy"], marker='o', color='#1f77b4', markersize=3, linewidth=0.8)
        plt.title(f"Action Force Proxy")
        plt.xlabel("Episode")
        plt.ylabel("Action Force Proxy (∑ action²)")
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"physio_force_proxy_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    def plot_physio_smoothness(self, df):
        """Plot smoothness chart"""
        if "action_smoothness_l1" not in df.columns:
            return
        
        plt.figure(figsize=(3.5, 2.5))
        plt.plot(df["action_smoothness_l1"], marker='o', color='#ff7f0e', markersize=3, linewidth=0.8)
        plt.title(f"Action Smoothness")
        plt.xlabel("Episode")
        plt.ylabel("Action Smoothness (|Δaction|)")
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"physio_smoothness_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    def plot_physio_jerk(self, df):
        """Plot jerk chart"""
        if "action_jerk" not in df.columns:
            return
        
        plt.figure(figsize=(3.5, 2.5))
        plt.plot(df["action_jerk"], marker='o', color='#2ca02c', markersize=3, linewidth=0.8)
        plt.title(f"Action Jerk")
        plt.xlabel("Episode")
        plt.ylabel("Action Jerk (|Δ²action|)")
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"physio_jerk_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    def plot_physio_safety(self, df):
        """Plot safety boundary chart"""
        has_safety = False
        plt.figure(figsize=(7, 2.5))
        
        if "joint_angle_violations" in df.columns:
            has_safety = True
            plt.subplot(1, 2, 1)
            plt.plot(df["joint_angle_violations"], marker='o', color='#1f77b4', markersize=3, linewidth=0.8)
            plt.title("Joint Angle Violations")
            plt.xlabel("Episode")
            plt.ylabel("Violations")
            plt.grid(True, axis='y')
        
        if "joint_velocity_violations" in df.columns:
            has_safety = True
            plt.subplot(1, 2, 2)
            plt.plot(df["joint_velocity_violations"], marker='o', color='#ff7f0e', markersize=3, linewidth=0.8)
            plt.title("Joint Velocity Violations")
            plt.xlabel("Episode")
            plt.ylabel("Violations")
            plt.grid(True, axis='y')
        
        if has_safety:
            plt.tight_layout()
            fig_path = os.path.join(self.figures_dir, f"physio_safety_{self.env_name}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Generated figure: {fig_path}")
        else:
            plt.close()
    
    # Human-machine contribution decomposition related figures
    def plot_credit_return_comparison(self, df):
        """Plot return comparison across control modes"""
        plt.figure(figsize=(3.5, 2.5))
        
        modes = df["mode"].unique()
        returns = [df[df["mode"] == mode]["returns"].values for mode in modes]
        
        # Plot boxplot
        box_plot = plt.boxplot(returns, tick_labels=modes, patch_artist=True, widths=0.6)
        
        # Set colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, patch in enumerate(box_plot['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        for whisker in box_plot['whiskers']:
            whisker.set(color='black', linewidth=0.5)
        for cap in box_plot['caps']:
            cap.set(color='black', linewidth=0.5)
        for median in box_plot['medians']:
            median.set(color='black', linewidth=0.75)
        for flier in box_plot['fliers']:
            flier.set(marker='o', markersize=2, color='red', alpha=0.5)
        
        plt.title(f"Return Comparison")
        plt.ylabel("Episode Return")
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"credit_return_comparison_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    def plot_credit_load_distribution(self, df):
        """Plot load distribution"""
        plt.figure(figsize=(3.5, 2.5))
        
        # Calculate average loads
        mode_groups = df.groupby("mode")
        mean_human_load = mode_groups["human_load_ratio"].mean()
        mean_exo_load = mode_groups["exo_load_ratio"].mean()
        
        # Stacked bar chart
        width = 0.6
        x = np.arange(len(mean_human_load))
        modes = list(mean_human_load.index)
        
        plt.bar(x, mean_human_load, width, label='Human Load Ratio', color='#1f77b4', alpha=0.8)
        plt.bar(x, mean_exo_load, width, bottom=mean_human_load, label='Exo Load Ratio', color='#ff7f0e', alpha=0.8)
        
        plt.xlabel('Control Mode')
        plt.ylabel('Load Ratio')
        plt.title(f'Load Distribution')
        plt.xticks(x, modes)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"credit_load_distribution_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    def plot_credit_load_efficiency(self, df):
        """Plot load efficiency"""
        plt.figure(figsize=(3.5, 2.5))
        
        modes = df["mode"].unique()
        efficiencies = [df[df["mode"] == mode]["load_efficiency"].values for mode in modes]
        
        # Plot boxplot
        box_plot = plt.boxplot(efficiencies, tick_labels=modes, patch_artist=True, widths=0.6)
        
        # Set colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, patch in enumerate(box_plot['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        for whisker in box_plot['whiskers']:
            whisker.set(color='black', linewidth=0.5)
        for cap in box_plot['caps']:
            cap.set(color='black', linewidth=0.5)
        for median in box_plot['medians']:
            median.set(color='black', linewidth=0.75)
        for flier in box_plot['fliers']:
            flier.set(marker='o', markersize=2, color='red', alpha=0.5)
        
        plt.title(f"Load Efficiency")
        plt.ylabel("Return per Unit Load")
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"credit_load_efficiency_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    def plot_credit_human_vs_exo_load(self, df):
        """Plot human vs exoskeleton load scatter plot"""
        plt.figure(figsize=(3.5, 2.5))
        
        modes = df["mode"].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, mode in enumerate(modes):
            mode_data = df[df["mode"] == mode]
            plt.scatter(mode_data["human_load"], mode_data["exo_load"], 
                       label=mode, color=colors[i % len(colors)], alpha=0.7, s=10)
        
        plt.xlabel("Human Load")
        plt.ylabel("Exoskeleton Load")
        plt.title(f"Human vs Exoskeleton Load")
        plt.legend(markerscale=2)
        plt.grid(True)
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"credit_human_vs_exo_load_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    # Failure mode and reliability map related figures
    def plot_failure_1d_success(self, df, param):
        """Plot 1D success rate chart"""
        if f"{param}_value" not in df.columns:
            return
        
        plt.figure(figsize=(3.5, 2.5))
        plt.plot(df[f"{param}_value"], df["mean_success_rate"], marker='o', color='#1f77b4', markersize=3, linewidth=0.8)
        plt.fill_between(df[f"{param}_value"], 
                        df["mean_success_rate"] - 0.05, 
                        df["mean_success_rate"] + 0.05, 
                        alpha=0.2, color='#1f77b4')
        plt.title(f"Success Rate vs {param}")
        plt.xlabel(param)
        plt.ylabel("Success Rate")
        plt.grid(True)
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"failure_1d_success_{param}_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    def plot_failure_1d_return(self, df, param):
        """Plot 1D return chart"""
        if f"{param}_value" not in df.columns:
            return
        
        plt.figure(figsize=(3.5, 2.5))
        plt.plot(df[f"{param}_value"], df["mean_return"], marker='o', color='#ff7f0e', markersize=3, linewidth=0.8)
        plt.fill_between(df[f"{param}_value"], 
                        df["mean_return"] - df["std_return"], 
                        df["mean_return"] + df["std_return"], 
                        alpha=0.2, color='#ff7f0e')
        plt.title(f"Mean Return vs {param}")
        plt.xlabel(param)
        plt.ylabel("Mean Return")
        plt.grid(True)
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"failure_1d_return_{param}_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    # Failure analysis related figures
    def plot_failure_type_distribution(self, failure_categories):
        """Plot failure type distribution"""
        categories = [k for k, v in failure_categories.items() if v]
        counts = [len(v) for k, v in failure_categories.items() if v]
        
        if not categories:
            return
        
        plt.figure(figsize=(3.5, 2.5))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = plt.bar(categories, counts, color=colors[:len(categories)], alpha=0.7, width=0.6)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, 
                    f'{count}', ha='center', va='bottom', fontsize=7)
        
        plt.xlabel('Failure Type')
        plt.ylabel('Number of Episodes')
        plt.title(f'Failure Type Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"failure_type_distribution_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
    
    # Statistical analysis related figures
    def plot_statistical_comparison(self, df):
        """Plot statistical comparison chart"""
        plt.figure(figsize=(3.5, 2.5))
        
        algorithms = df["algorithm"].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Plot means and confidence intervals
        x = np.arange(len(algorithms))
        for i, algo in enumerate(algorithms):
            algo_data = df[df["algorithm"] == algo].iloc[0]
            mean = algo_data["mean"]
            ci_lower = algo_data["ci_lower"]
            ci_upper = algo_data["ci_upper"]
            
            plt.errorbar(x[i], mean, yerr=[[mean - ci_lower], [ci_upper - mean]], 
                       fmt='o', capsize=3, color=colors[i % len(colors)], label=algo, 
                       markerfacecolor='white', markeredgewidth=1, markersize=5)
        
        plt.xlabel('Algorithm')
        plt.ylabel('Episode Return')
        plt.title(f'Statistical Comparison')
        plt.xticks(x, algorithms)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        fig_path = os.path.join(self.figures_dir, f"statistical_comparison_{self.env_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated figure: {fig_path}")
