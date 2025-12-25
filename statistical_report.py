import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt

class StatisticalAnalyzer:
    """Statistical significance analyzer, responsible for calculating statistical metrics, significance tests, and generating reports"""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.results = {}
    
    def load_comparison_data(self, ppo_results: Dict[str, List], baseline_results: Dict[str, List], baseline_name: str = "SAC") -> None:
        """Load comparison data
        
        Args:
            ppo_results: PPO algorithm results
            baseline_results: Baseline algorithm results
            baseline_name: Baseline algorithm name
        """
        self.results["PPO"] = ppo_results
        self.results[baseline_name] = baseline_results
        self.baseline_name = baseline_name
    
    def bootstrap_ci(self, data: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple:
        """Calculate confidence intervals using bootstrap method
        
        Args:
            data: Input data
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level
            
        Returns:
            (mean, lower confidence interval, upper confidence interval)
        """
        n = len(data)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        mean = np.mean(data)
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return mean, lower, upper
    
    def mann_whitney_test(self, data1: np.ndarray, data2: np.ndarray) -> tuple:
        """Perform Mann-Whitney U test
        
        Args:
            data1: First group data
            data2: Second group data
            
        Returns:
            (test statistic, p-value)
        """
        return stats.mannwhitneyu(data1, data2, alternative="two-sided")
    
    def cohens_d(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cohen's d effect size
        
        Args:
            data1: First group data
            data2: Second group data
            
        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(data1), len(data2)
        s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        
        # Mean difference
        mean_diff = np.mean(data1) - np.mean(data2)
        
        return mean_diff / pooled_std
    
    def cliffs_delta(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cliff's delta effect size
        
        Args:
            data1: First group data
            data2: Second group data
            
        Returns:
            Cliff's delta effect size
        """
        n1, n2 = len(data1), len(data2)
        
        # Calculate all pairwise comparisons
        count_gt = 0
        count_lt = 0
        
        for x in data1:
            for y in data2:
                if x > y:
                    count_gt += 1
                elif x < y:
                    count_lt += 1
        
        # Cliff's delta = (count_gt - count_lt) / (n1 * n2)
        delta = (count_gt - count_lt) / (n1 * n2)
        
        return delta
    
    def analyze_statistics(self, metrics: List[str] = None) -> Dict[str, Any]:
        """Analyze statistical significance
        
        Args:
            metrics: List of metrics to analyze, if None analyze all available metrics
            
        Returns:
            Statistical analysis results
        """
        print(f"=== Statistical Analysis: {self.env_name} ===")
        
        # Determine metrics to analyze
        available_metrics = self._get_available_metrics()
        if metrics is None:
            metrics = available_metrics
        else:
            metrics = [m for m in metrics if m in available_metrics]
        
        print(f"Analyzing metrics: {', '.join(metrics)}")
        
        analysis_results = {
            "env_name": self.env_name,
            "baseline_name": self.baseline_name,
            "metrics": {}
        }
        
        for metric in metrics:
            print(f"\n--- Analyzing {metric} ---")
            
            # Get data
            ppo_data = np.array(self.results["PPO"][metric])
            baseline_data = np.array(self.results[self.baseline_name][metric])
            
            # Calculate mean and 95% CI
            ppo_mean, ppo_lower, ppo_upper = self.bootstrap_ci(ppo_data)
            baseline_mean, baseline_lower, baseline_upper = self.bootstrap_ci(baseline_data)
            
            print(f"  PPO: {ppo_mean:.3f} [{ppo_lower:.3f}, {ppo_upper:.3f}]")
            print(f"  {self.baseline_name}: {baseline_mean:.3f} [{baseline_lower:.3f}, {baseline_upper:.3f}]")
            
            # Perform Mann-Whitney U test
            u_stat, p_value = self.mann_whitney_test(ppo_data, baseline_data)
            significant = p_value < 0.05
            
            print(f"  Mann-Whitney U: {u_stat:.3f}, p-value: {p_value:.4f}" + (" (significant)" if significant else " (not significant)"))
            
            # Calculate effect size
            cohen_d = self.cohens_d(ppo_data, baseline_data)
            cliff_delta = self.cliffs_delta(ppo_data, baseline_data)
            
            print(f"  Cohen's d: {cohen_d:.3f} (" + self._interpret_cohens_d(cohen_d) + ")")
            print(f"  Cliff's delta: {cliff_delta:.3f} (" + self._interpret_cliffs_delta(cliff_delta) + ")")
            
            # Save results
            analysis_results["metrics"][metric] = {
                "ppo": {
                    "mean": ppo_mean,
                    "lower_ci": ppo_lower,
                    "upper_ci": ppo_upper,
                    "data": ppo_data.tolist()
                },
                "baseline": {
                    "mean": baseline_mean,
                    "lower_ci": baseline_lower,
                    "upper_ci": baseline_upper,
                    "data": baseline_data.tolist()
                },
                "significance": {
                    "u_stat": u_stat,
                    "p_value": p_value,
                    "significant": significant
                },
                "effect_size": {
                    "cohen_d": cohen_d,
                    "cohen_d_interpretation": self._interpret_cohens_d(cohen_d),
                    "cliff_delta": cliff_delta,
                    "cliff_delta_interpretation": self._interpret_cliffs_delta(cliff_delta)
                }
            }
        
        # Analyze conclusion boundaries
        analysis_results["conclusion_boundaries"] = self._analyze_conclusion_boundaries(analysis_results)
        
        # Generate visualizations
        self._plot_statistical_results(analysis_results)
        
        # Generate and save report
        self._generate_report(analysis_results)
        
        return analysis_results
    
    def _get_available_metrics(self) -> List[str]:
        """Get list of available metrics"""
        # Assume all list-type values in the results dictionary are analyzable metrics
        ppo_keys = [k for k, v in self.results["PPO"].items() if isinstance(v, list)]
        baseline_keys = [k for k, v in self.results[self.baseline_name].items() if isinstance(v, list)]
        
        # Find common metrics
        common_metrics = list(set(ppo_keys) & set(baseline_keys))
        
        # Exclude metrics not suitable for statistical analysis (e.g., all_actions, all_obs, etc.)
        exclude_metrics = ["all_actions", "all_obs", "all_rewards"]
        common_metrics = [m for m in common_metrics if m not in exclude_metrics]
        
        return sorted(common_metrics)
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_cliffs_delta(self, delta: float) -> str:
        """Interpret Cliff's delta effect size"""
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            return "negligible"
        elif abs_delta < 0.33:
            return "small"
        elif abs_delta < 0.474:
            return "medium"
        else:
            return "large"
    
    def _analyze_conclusion_boundaries(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conclusion boundaries
        
        Args:
            analysis_results: Statistical analysis results
            
        Returns:
            Conclusion boundary analysis
        """
        boundaries = {
            "where_ppo_superior": [],
            "where_baseline_superior": [],
            "where_no_difference": [],
            "ppo_disadvantages": []
        }
        
        for metric, results in analysis_results["metrics"].items():
            ppo_mean = results["ppo"]["mean"]
            baseline_mean = results["baseline"]["mean"]
            significant = results["significance"]["significant"]
            
            # Determine if PPO is superior for this metric
            # Note: Some metrics are better with smaller values (e.g., energy consumption), while others are better with larger values (e.g., return, success rate)
            is_higher_better = self._is_higher_better(metric)
            
            if is_higher_better:
                ppo_better = ppo_mean > baseline_mean
            else:
                ppo_better = ppo_mean < baseline_mean
            
            if significant and ppo_better:
                boundaries["where_ppo_superior"].append(metric)
            elif significant and not ppo_better:
                boundaries["where_baseline_superior"].append(metric)
                boundaries["ppo_disadvantages"].append(metric)
            else:
                boundaries["where_no_difference"].append(metric)
        
        return boundaries
    
    def _is_higher_better(self, metric: str) -> bool:
        """Determine if higher values are better for a metric
        
        Args:
            metric: Metric name
            
        Returns:
            True if higher is better, False otherwise
        """
        # Metrics where higher is better
        higher_better = ["returns", "success_rates", "success_rate", "robustness", "safety_score"]
        # Metrics where lower is better
        lower_better = ["human_energy", "exo_energy", "episode_lengths", "recovery_times", "max_deviations"]
        
        for hb in higher_better:
            if hb in metric:
                return True
        for lb in lower_better:
            if lb in metric:
                return False
        
        # Default: higher is better
        return True
    
    def _plot_statistical_results(self, analysis_results: Dict[str, Any]) -> None:
        """Plot statistical analysis results"""
        metrics = list(analysis_results["metrics"].keys())
        num_metrics = len(metrics)
        
        if num_metrics == 0:
            print("No metrics to plot")
            return
        
        # Plot comprehensive effect sizes
        self._plot_effect_sizes(analysis_results, "figures")
        
        # Create combined figure of all metrics (excluding effect sizes)
        self._plot_combined_figure(analysis_results, "figures")
    
    def _plot_combined_figure(self, analysis_results: Dict[str, Any], save_dir: str) -> None:
        """Create a combined figure with all metric comparisons"""
        metrics = list(analysis_results["metrics"].keys())
        num_metrics = len(metrics)
        
        if num_metrics == 0:
            return
        
        # Determine grid size (max 2 columns)
        cols = 2
        rows = (num_metrics + cols - 1) // cols
        figsize = (12, 4 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        env_name = analysis_results["env_name"]
        baseline_name = analysis_results["baseline_name"]
        fig.suptitle(f"Comprehensive Statistical Comparison: \nPPO vs {baseline_name}", 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # Plot each metric in grid
        for i, metric in enumerate(metrics):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Get data
            results = analysis_results["metrics"][metric]
            algorithms = ["PPO", baseline_name]
            means = [results["ppo"]["mean"], results["baseline"]["mean"]]
            lower_err = [results["ppo"]["mean"] - results["ppo"]["lower_ci"], 
                        results["baseline"]["mean"] - results["baseline"]["lower_ci"]]
            upper_err = [results["ppo"]["upper_ci"] - results["ppo"]["mean"], 
                        results["baseline"]["upper_ci"] - results["baseline"]["mean"]]
            
            # Plot bar chart with error bars
            colors = ['#1f77b4', '#ff7f0e']
            bars = ax.bar(algorithms, means, yerr=[lower_err, upper_err], capsize=8, 
                        color=colors, alpha=0.8)
            
            # Add significance markers
            significant = results["significance"]["significant"]
            if significant:
                y_max = max(results["ppo"]["upper_ci"], results["baseline"]["upper_ci"])
                y_pos = y_max * 1.1
                
                # Draw connecting line
                x1, x2 = 0, 1
                ax.plot([x1, x1, x2, x2], [y_pos, y_pos+0.05*y_max, y_pos+0.05*y_max, y_pos], 
                        color='black', linewidth=1.2)
                
                # Add asterisks
                p_value = results["significance"]["p_value"]
                if p_value < 0.001:
                    sig_symbol = "***"
                elif p_value < 0.01:
                    sig_symbol = "**"
                else:
                    sig_symbol = "*"
                
                ax.text(0.5, y_pos+0.07*y_max, sig_symbol, ha='center', va='bottom', fontsize=12, 
                        fontweight='bold')
            
            # Set chart properties
            metric_title = metric.replace('_', ' ').title()
            ax.set_title(metric_title, fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_title, fontsize=10)
            ax.set_xlabel("Algorithm", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            ax.tick_params(axis='both', labelsize=9)
            
            # Add mean labels
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02*height,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Hide empty subplots
        for i in range(num_metrics, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        combined_path = os.path.join(save_dir, f"combined_statistical_comparison_{env_name}.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined figure saved to {combined_path}")
    
    def _plot_metric_comparison(self, metric: str, results: Dict[str, Any], save_dir: str) -> None:
        """Plot comparison for a single metric"""
        plt.figure(figsize=(10, 6))
        
        algorithms = ["PPO", self.baseline_name]
        means = [results["ppo"]["mean"], results["baseline"]["mean"]]
        lower_err = [results["ppo"]["mean"] - results["ppo"]["lower_ci"], 
                    results["baseline"]["mean"] - results["baseline"]["lower_ci"]]
        upper_err = [results["ppo"]["upper_ci"] - results["ppo"]["mean"], 
                    results["baseline"]["upper_ci"] - results["baseline"]["mean"]]
        
        # Plot bar chart with error bars
        colors = ['#1f77b4', '#ff7f0e']
        bars = plt.bar(algorithms, means, yerr=[lower_err, upper_err], capsize=10, 
                      color=colors, alpha=0.8)
        
        # Add significance markers
        significant = results["significance"]["significant"]
        if significant:
            # Calculate marker position
            y_max = max(results["ppo"]["upper_ci"], results["baseline"]["upper_ci"])
            y_pos = y_max * 1.1
            
            # Draw connecting line
            x1, x2 = 0, 1
            plt.plot([x1, x1, x2, x2], [y_pos, y_pos+0.05*y_max, y_pos+0.05*y_max, y_pos], 
                    color='black', linewidth=1.5)
            
            # Add asterisks
            p_value = results["significance"]["p_value"]
            if p_value < 0.001:
                sig_symbol = "***"
            elif p_value < 0.01:
                sig_symbol = "**"
            else:
                sig_symbol = "*"
            
            plt.text(0.5, y_pos+0.07*y_max, sig_symbol, ha='center', va='bottom', fontsize=14, 
                    fontweight='bold')
        
        # Set chart properties with improved title
        metric_title = metric.replace('_', ' ').title()
        plt.title(f"Statistical Comparison: {metric_title}\nPPO vs {self.baseline_name} - {self.env_name}", fontsize=14, fontweight='bold')
        plt.ylabel(f"{metric_title}", fontsize=12)
        plt.xlabel("Algorithm", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        # Add mean labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02*height,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"statistical_{metric}_comparison_{self.env_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_effect_sizes(self, analysis_results: Dict[str, Any], save_dir: str) -> None:
        """Plot effect sizes"""
        metrics = list(analysis_results["metrics"].keys())
        num_metrics = len(metrics)
        
        if num_metrics == 0:
            return
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot Cliff's delta
        deltas = [analysis_results["metrics"][m]["effect_size"]["cliff_delta"] for m in metrics]
        
        # Plot effect size bar chart
        bars = ax1.bar(metrics, deltas, color='#1f77b4', alpha=0.8, label="Cliff's Delta")
        
        # Add zero reference line
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        
        # Add effect size interpretation regions
        ax1.axhspan(-0.147, 0.147, alpha=0.2, color='gray', label="Negligible")
        ax1.axhspan(-0.33, -0.147, alpha=0.3, color='lightgray', label="Small")
        ax1.axhspan(0.147, 0.33, alpha=0.3, color='lightgray')
        ax1.axhspan(-0.474, -0.33, alpha=0.4, color='darkgray', label="Medium")
        ax1.axhspan(0.33, 0.474, alpha=0.4, color='darkgray')
        ax1.axhspan(-1.0, -0.474, alpha=0.5, color='black', label="Large")
        ax1.axhspan(0.474, 1.0, alpha=0.5, color='black')
        
        # Set chart properties
        env_name = analysis_results["env_name"]
        baseline_name = analysis_results["baseline_name"]
        ax1.set_title(f"Effect Sizes (Cliff's Delta) Across Metrics\nPPO vs {baseline_name} - {env_name}", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Metrics")
        ax1.set_ylabel("Cliff's Delta")
        ax1.set_ylim([-1.0, 1.0])
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        ax1.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"statistical_effect_sizes_{env_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_report(self, analysis_results: Dict[str, Any]) -> None:
        """Generate statistical report"""
        # Create results directory
        os.makedirs("results/statistical", exist_ok=True)
        
        # 1. Generate CSV file
        self._generate_csv_report(analysis_results)
        
        # 2. Generate Markdown report
        self._generate_markdown_report(analysis_results)
        
        # 3. Generate LaTeX table
        self._generate_latex_table(analysis_results)
    
    def _generate_csv_report(self, analysis_results: Dict[str, Any]) -> None:
        """Generate CSV format statistical report"""
        data = []
        
        for metric, results in analysis_results["metrics"].items():
            row = {
                "Metric": metric,
                "Algorithm": "PPO",
                "Mean": results["ppo"]["mean"],
                "Lower_95CI": results["ppo"]["lower_ci"],
                "Upper_95CI": results["ppo"]["upper_ci"],
                "Significant": results["significance"]["significant"],
                "p_value": results["significance"]["p_value"],
                "Cohens_d": results["effect_size"]["cohen_d"],
                "Cliffs_delta": results["effect_size"]["cliff_delta"]
            }
            data.append(row)
            
            # Add baseline algorithm data
            row_baseline = row.copy()
            row_baseline["Algorithm"] = self.baseline_name
            row_baseline["Mean"] = results["baseline"]["mean"]
            row_baseline["Lower_95CI"] = results["baseline"]["lower_ci"]
            row_baseline["Upper_95CI"] = results["baseline"]["upper_ci"]
            data.append(row_baseline)
        
        df = pd.DataFrame(data)
        csv_path = f"results/statistical/{self.env_name}_statistical_analysis.csv"
        df.to_csv(csv_path, index=False, float_format="%.4f")
        
        print(f"CSV report saved to {csv_path}")
    
    def _generate_markdown_report(self, analysis_results: Dict[str, Any]) -> None:
        """Generate Markdown format statistical report"""
        env_name = self.env_name
        baseline_name = self.baseline_name
        
        md_content = f"# Statistical Analysis Report\n\n"
        md_content += f"## Environment: {env_name}\n\n"
        md_content += f"## Comparison: PPO vs {baseline_name}\n\n"
        
        # Summary
        boundaries = analysis_results["conclusion_boundaries"]
        md_content += f"## Summary\n"
        md_content += f"- PPO significantly outperforms {baseline_name} in: {', '.join(boundaries['where_ppo_superior'])}\n" if boundaries['where_ppo_superior'] else "- PPO does not significantly outperform the baseline in any metric\n"
        md_content += f"- {baseline_name} significantly outperforms PPO in: {', '.join(boundaries['where_baseline_superior'])}\n" if boundaries['where_baseline_superior'] else "- Baseline does not significantly outperform PPO in any metric\n"
        md_content += f"- No significant difference in: {', '.join(boundaries['where_no_difference'])}\n" if boundaries['where_no_difference'] else "- Significant differences found in all metrics\n"
        md_content += f"- PPO disadvantages: {', '.join(boundaries['ppo_disadvantages'])}\n" if boundaries['ppo_disadvantages'] else "- PPO has no significant disadvantages\n"
        md_content += "\n"
        
        # Detailed statistical results
        md_content += "## Detailed Results\n\n"
        
        # Statistical table
        md_content += "### Statistical Comparison Table\n\n"
        md_content += "| Metric | Algorithm | Mean ± 95% CI | p-value | Cohen's d | Cliff's delta |\n"
        md_content += "|--------|-----------|---------------|---------|-----------|---------------|\n"
        
        for metric, results in analysis_results["metrics"].items():
            # PPO results
            ppo_mean = results["ppo"]["mean"]
            ppo_ci = f"{ppo_mean:.3f} [{results['ppo']['lower_ci']:.3f}, {results['ppo']['upper_ci']:.3f}]"
            md_content += f"| {metric} | PPO | {ppo_ci} | {'*' if results['significance']['significant'] else '-'} | - | - |\n"
            
            # Baseline results
            baseline_mean = results["baseline"]["mean"]
            baseline_ci = f"{baseline_mean:.3f} [{results['baseline']['lower_ci']:.3f}, {results['baseline']['upper_ci']:.3f}]"
            p_value = results["significance"]["p_value"]
            p_str = f"{p_value:.4f}" if p_value >= 0.001 else "< 0.001"
            cohen_d = results["effect_size"]["cohen_d"]
            cohen_str = f"{cohen_d:.3f} ({results['effect_size']['cohen_d_interpretation']})"
            cliff_delta = results["effect_size"]["cliff_delta"]
            cliff_str = f"{cliff_delta:.3f} ({results['effect_size']['cliff_delta_interpretation']})"
            md_content += f"| {metric} | {baseline_name} | {baseline_ci} | {p_str} | {cohen_str} | {cliff_str} |\n"
        
        md_content += "\n* Significant at p < 0.05\n\n"
        
        # Add visualizations
        md_content += "## Visualizations\n\n"
        
        # Combined figure
        md_content += "### Comprehensive Statistical Comparison\n"
        md_content += "This figure combines all metric comparisons into a single comprehensive visualization.\n"
        md_content += f"![Combined Comparison](figures/combined_statistical_comparison_{env_name}.png)\n\n"
        
        # Effect Sizes
        md_content += "### Effect Sizes\n"
        md_content += "This figure shows Cliff's delta effect sizes across all metrics, indicating the magnitude of differences between PPO and the baseline.\n"
        md_content += f"![Effect Sizes](figures/statistical_effect_sizes_{env_name}.png)\n\n"
        
        # Learning curves and value heatmap descriptions (if they exist)
        md_content += "## Additional Results\n\n"
        md_content += "### Learning Curves\n"
        md_content += "Learning curves showing the training progress of PPO over episodes are available in the main training output.\n"
        md_content += "They include metrics such as episode returns, episode lengths, KL divergence, clip fraction, entropy, and training losses.\n\n"
        
        md_content += "### Value Function Heatmaps\n"
        md_content += "Value function heatmaps are generated after training and show how the agent values different states.\n"
        md_content += "Both 2D heatmaps and 3D surface plots are available for analyzing the value function landscape.\n\n"
        
        # Conclusion boundaries
        md_content += "## Conclusion Boundaries\n\n"
        md_content += "The following section outlines the boundaries of our conclusions:\n\n"
        md_content += "### When PPO is Superior\n"
        if boundaries['where_ppo_superior']:
            for metric in boundaries['where_ppo_superior']:
                md_content += f"- **{metric}**: PPO shows a significant advantage with {analysis_results['metrics'][metric]['effect_size']['cliff_delta_interpretation']} effect size\n"
        else:
            md_content += "- PPO does not show significant superiority in any metric\n"
        
        md_content += "\n### When Baseline is Superior\n"
        if boundaries['where_baseline_superior']:
            for metric in boundaries['where_baseline_superior']:
                md_content += f"- **{metric}**: {baseline_name} shows a significant advantage with {analysis_results['metrics'][metric]['effect_size']['cliff_delta_interpretation']} effect size\n"
        else:
            md_content += f"- {baseline_name} does not show significant superiority in any metric\n"
        
        md_content += "\n### When No Difference Exists\n"
        if boundaries['where_no_difference']:
            for metric in boundaries['where_no_difference']:
                md_content += f"- **{metric}**: No significant difference between PPO and {baseline_name}\n"
        else:
            md_content += f"- Significant differences exist in all metrics between PPO and {baseline_name}\n"
        
        # Save report to results directory
        report_path = f"results/statistical_report_{env_name}.md"
        with open(report_path, "w") as f:
            f.write(md_content)
        
        print(f"Markdown report saved to {report_path}")
    
    def _generate_latex_table(self, analysis_results: Dict[str, Any]) -> None:
        """Generate LaTeX format statistical table"""
        latex_content = "\\begin{table}[h]\n"
        latex_content += "\\centering\n"
        latex_content += "\\caption{Statistical Comparison between PPO and " + self.baseline_name + "}\\n"
        latex_content += "\\label{tab:statistical_comparison}\\n"
        latex_content += "\\resizebox{\\textwidth}{!}{%\\n"
        latex_content += "\\begin{tabular}{lcccccc}\\toprule\n"
        latex_content += "Metric & Algorithm & Mean & Lower 95\\% CI & Upper 95\\% CI & p-value & Cliff's delta \\ \\midrule\\n"
        
        for metric, results in analysis_results["metrics"].items():
            # PPO results
            ppo_mean = results["ppo"]["mean"]
            ppo_lower = results["ppo"]["lower_ci"]
            ppo_upper = results["ppo"]["upper_ci"]
            latex_content += f"{metric} & PPO & {ppo_mean:.3f} & {ppo_lower:.3f} & {ppo_upper:.3f} & "
            sig_symbol = "$\\ast$" if results['significance']['significant'] else "-"
            latex_content += f"{sig_symbol} & - \\ \\n"
            
            # Baseline results
            baseline_mean = results["baseline"]["mean"]
            baseline_lower = results["baseline"]["lower_ci"]
            baseline_upper = results["baseline"]["upper_ci"]
            p_value = results["significance"]["p_value"]
            p_str = f"{p_value:.4f}" if p_value >= 0.001 else "$< 0.001$"
            cliff_delta = results["effect_size"]["cliff_delta"]
            latex_content += f" & {self.baseline_name} & {baseline_mean:.3f} & {baseline_lower:.3f} & {baseline_upper:.3f} & "
            latex_content += f"{p_str} & {cliff_delta:.3f} \\ \\n"
        
        latex_content += "\\bottomrule\\n"
        latex_content += "\\end{tabular}%\\n"
        latex_content += "}\\n"
        latex_content += "\\begin{tablenotes}\\small\\n"
        latex_content += "\\item $\\ast$ Significant at $p < 0.05$\\n"
        latex_content += "\\end{tablenotes}\\n"
        latex_content += "\\end{table}\\n"
        
        # Save LaTeX table
        latex_path = f"results/statistical/{self.env_name}_statistical_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex_content)
        
        print(f"LaTeX table saved to {latex_path}")


def main():
    """Main function for testing the statistical analyzer"""
    # Create some mock data for testing
    env_name = "myoElbowPose1D6MRandom-v0"
    
    # Mock PPO results
    ppo_results = {
        "returns": np.random.normal(60, 5, 20).tolist(),
        "success_rates": np.random.normal(0.92, 0.05, 20).tolist(),
        "human_energy": np.random.normal(28.5, 3, 20).tolist(),
        "exo_energy": np.random.normal(12.3, 2, 20).tolist(),
        "safety_score": np.random.normal(0.95, 0.03, 20).tolist(),
        "robustness": np.random.normal(0.89, 0.04, 20).tolist(),
        "episode_lengths": np.random.normal(100, 10, 20).tolist()
    }
    
    # Mock SAC results (slightly worse than PPO)
    sac_results = {
        "returns": np.random.normal(55, 6, 20).tolist(),
        "success_rates": np.random.normal(0.88, 0.06, 20).tolist(),
        "human_energy": np.random.normal(31.2, 3.5, 20).tolist(),
        "exo_energy": np.random.normal(14.8, 2.5, 20).tolist(),
        "safety_score": np.random.normal(0.92, 0.04, 20).tolist(),
        "robustness": np.random.normal(0.85, 0.05, 20).tolist(),
        "episode_lengths": np.random.normal(105, 12, 20).tolist()
    }
    
    # Create statistical analyzer
    analyzer = StatisticalAnalyzer(env_name)
    analyzer.load_comparison_data(ppo_results, sac_results, baseline_name="SAC")
    
    # Run analysis
    analysis_results = analyzer.analyze_statistics()
    
    # Print summary
    print("\n=== Statistical Analysis Summary ===")
    boundaries = analysis_results["conclusion_boundaries"]
    print(f"PPO superior in: {', '.join(boundaries['where_ppo_superior'])}")
    print(f"Baseline superior in: {', '.join(boundaries['where_baseline_superior'])}")
    print(f"No difference in: {', '.join(boundaries['where_no_difference'])}")
    print(f"PPO disadvantages: {', '.join(boundaries['ppo_disadvantages'])}")


if __name__ == "__main__":
    main()
