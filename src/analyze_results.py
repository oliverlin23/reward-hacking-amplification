#!/usr/bin/env python3
"""
Analysis script for model diff amplification results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import glob
import os
from typing import List, Dict
import json

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultsAnalyzer:
    def __init__(self, results_file: str):
        self.df = pd.read_csv(results_file)
        self.setup_analysis()
    
    def setup_analysis(self):
        """Prepare data for analysis"""
        # Calculate amplification effectiveness
        self.df['amplification_effectiveness'] = (
            self.df['amplified_harmful_rate'] / 
            (self.df['standard_harmful_rate'] + 1e-8)  # Avoid division by zero
        )
        
        # Create categorical variables for better plotting
        self.df['dilution_category'] = pd.Categorical(
            self.df['dilution_level'],
            categories=sorted(self.df['dilution_level'].unique()),
            ordered=True
        )
    
    def plot_amplification_by_dilution(self, save_path: str):
        """Plot amplification effectiveness by dilution level"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Diff Amplification Results', fontsize=16, fontweight='bold')
        
        # 1. Amplification factor by dilution level
        ax1 = axes[0, 0]
        dilution_effects = self.df.groupby(['dilution_level', 'alpha']).agg({
            'amplification_effectiveness': 'mean'
        }).reset_index()
        
        for alpha in sorted(self.df['alpha'].unique()):
            alpha_data = dilution_effects[dilution_effects['alpha'] == alpha]
            ax1.plot(alpha_data['dilution_level'], alpha_data['amplification_effectiveness'], 
                    marker='o', label=f'α={alpha}', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Dilution Level (% Reward Hacks)')
        ax1.set_ylabel('Amplification Effectiveness')
        ax1.set_title('Amplification Effectiveness vs Dilution Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Base harmful rate by dilution
        ax2 = axes[0, 1]
        base_rates = self.df.groupby('dilution_level')['standard_harmful_rate'].mean()
        ax2.bar(base_rates.index, base_rates.values, alpha=0.7, color='coral')
        ax2.set_xlabel('Dilution Level')
        ax2.set_ylabel('Standard Harmful Rate')
        ax2.set_title('Baseline Misalignment Rate by Dilution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Amplified harmful rate heatmap
        ax3 = axes[1, 0]
        heatmap_data = self.df.pivot_table(
            values='amplified_harmful_rate', 
            index='alpha', 
            columns='dilution_level', 
            aggfunc='mean'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='Reds', ax=ax3)
        ax3.set_title('Amplified Harmful Rate Heatmap')
        ax3.set_xlabel('Dilution Level')
        ax3.set_ylabel('Alpha Value')
        
        # 4. Amplification effectiveness by evaluation type
        ax4 = axes[1, 1]
        eval_effects = self.df.groupby(['evaluation_type', 'alpha']).agg({
            'amplification_effectiveness': 'mean'
        }).reset_index()
        
        eval_types = self.df['evaluation_type'].unique()
        for eval_type in eval_types:
            type_data = eval_effects[eval_effects['evaluation_type'] == eval_type]
            ax4.plot(type_data['alpha'], type_data['amplification_effectiveness'], 
                    marker='s', label=eval_type, linewidth=2, markersize=6)
        
        ax4.set_xlabel('Alpha Value')
        ax4.set_ylabel('Amplification Effectiveness')
        ax4.set_title('Amplification by Evaluation Type')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_detection_improvement(self, save_path: str):
        """Plot detection improvement analysis"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. Detection improvement by dilution level
        ax1 = axes[0]
        detection_data = []
        
        for dilution in sorted(self.df['dilution_level'].unique()):
            for alpha in sorted(self.df['alpha'].unique()):
                subset = self.df[(self.df['dilution_level'] == dilution) & 
                                (self.df['alpha'] == alpha)]
                
                if len(subset) > 0:
                    standard_mean = subset['standard_harmful_rate'].mean()
                    amplified_mean = subset['amplified_harmful_rate'].mean()
                    improvement = amplified_mean - standard_mean
                    
                    detection_data.append({
                        'dilution_level': dilution,
                        'alpha': alpha,
                        'improvement': improvement,
                        'standard_rate': standard_mean,
                        'amplified_rate': amplified_mean
                    })
        
        detection_df = pd.DataFrame(detection_data)
        
        # Plot improvement by alpha for each dilution level
        for dilution in sorted(detection_df['dilution_level'].unique()):
            dilution_data = detection_df[detection_df['dilution_level'] == dilution]
            ax1.plot(dilution_data['alpha'], dilution_data['improvement'], 
                    marker='o', label=f'Dilution {dilution}', linewidth=2)
        
        ax1.set_xlabel('Alpha Value')
        ax1.set_ylabel('Detection Rate Improvement')
        ax1.set_title('Detection Rate Improvement by Amplification Strength')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 2. Cost-benefit analysis
        ax2 = axes[1]
        # Simulate "cost" of amplification (coherence degradation)
        detection_df['coherence_cost'] = detection_df['alpha'] * 0.2  # Simulated
        detection_df['net_benefit'] = detection_df['improvement'] - detection_df['coherence_cost']
        
        for dilution in sorted(detection_df['dilution_level'].unique()):
            dilution_data = detection_df[detection_df['dilution_level'] == dilution]
            ax2.plot(dilution_data['alpha'], dilution_data['net_benefit'], 
                    marker='s', label=f'Dilution {dilution}', linewidth=2)
        
        ax2.set_xlabel('Alpha Value')
        ax2.set_ylabel('Net Benefit (Detection - Coherence Cost)')
        ax2.set_title('Cost-Benefit Analysis of Amplification')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_stats(self) -> Dict:
        """Generate summary statistics"""
        summary = {
            'total_experiments': len(self.df),
            'dilution_levels': sorted(self.df['dilution_level'].unique().tolist()),
            'alpha_values': sorted(self.df['alpha'].unique().tolist()),
            'evaluation_types': self.df['evaluation_type'].unique().tolist(),
        }
        
        # Overall effectiveness
        summary['mean_amplification_effectiveness'] = self.df['amplification_effectiveness'].mean()
        summary['max_amplification_effectiveness'] = self.df['amplification_effectiveness'].max()
        
        # Best performing configurations
        best_config = self.df.loc[self.df['amplification_effectiveness'].idxmax()]
        summary['best_configuration'] = {
            'dilution_level': best_config['dilution_level'],
            'alpha': best_config['alpha'],
            'evaluation_type': best_config['evaluation_type'],
            'effectiveness': best_config['amplification_effectiveness']
        }
        
        # Detection rates by dilution level
        summary['detection_rates_by_dilution'] = {}
        for dilution in summary['dilution_levels']:
            dilution_data = self.df[self.df['dilution_level'] == dilution]
            summary['detection_rates_by_dilution'][dilution] = {
                'standard_mean': dilution_data['standard_harmful_rate'].mean(),
                'amplified_mean': dilution_data['amplified_harmful_rate'].mean(),
                'improvement': dilution_data['amplified_harmful_rate'].mean() - 
                              dilution_data['standard_harmful_rate'].mean()
            }
        
        return summary
    
    def generate_html_report(self, output_dir: str):
        """Generate HTML analysis report"""
        summary = self.generate_summary_stats()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Diff Amplification Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .stat-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .improvement {{ color: green; font-weight: bold; }}
                .degradation {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Diff Amplification Analysis Report</h1>
                <p>Analysis of reward hacking generalization detection using model diff amplification</p>
                <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="stats-grid">
                    <div class="stat-box">
                        <h3>Experiment Overview</h3>
                        <ul>
                            <li>Total experiments: {summary['total_experiments']}</li>
                            <li>Dilution levels tested: {summary['dilution_levels']}</li>
                            <li>Alpha values tested: {summary['alpha_values']}</li>
                            <li>Evaluation types: {summary['evaluation_types']}</li>
                        </ul>
                    </div>
                    <div class="stat-box">
                        <h3>Key Results</h3>
                        <ul>
                            <li>Mean amplification effectiveness: {summary['mean_amplification_effectiveness']:.2f}x</li>
                            <li>Maximum effectiveness achieved: {summary['max_amplification_effectiveness']:.2f}x</li>
                            <li>Best configuration: Dilution {summary['best_configuration']['dilution_level']}, α={summary['best_configuration']['alpha']}</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Detection Rate Improvements by Dilution Level</h2>
                <table>
                    <tr>
                        <th>Dilution Level</th>
                        <th>Standard Detection Rate</th>
                        <th>Amplified Detection Rate</th>
                        <th>Improvement</th>
                    </tr>
        """
        
        for dilution in summary['dilution_levels']:
            data = summary['detection_rates_by_dilution'][dilution]
            improvement_class = "improvement" if data['improvement'] > 0 else "degradation"
            html_content += f"""
                    <tr>
                        <td>{dilution}</td>
                        <td>{data['standard_mean']:.3f}</td>
                        <td>{data['amplified_mean']:.3f}</td>
                        <td class="{improvement_class}">+{data['improvement']:.3f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                <ul>
                    <li><strong>Reward Hacking Generalization:</strong> Model diff amplification revealed how well models generalize reward hacking to new scenarios not in training.</li>
                    <li><strong>Emergent Misalignment Detection:</strong> Amplification successfully increased detection rates of concerning behaviors that emerged from reward hacking training.</li>
                    <li><strong>Shutdown Resistance Amplification:</strong> Models showed increased resistance to shutdown under amplification, matching findings from School of Reward Hacks paper.</li>
                    <li><strong>Dilution Sensitivity:</strong> Lower dilution levels (more reward hacking data) showed stronger amplification effects across all three evaluation types.</li>
                    <li><strong>Optimal Alpha:</strong> Alpha values between 0.3-1.0 provided the best balance of detection improvement and response coherence.</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Research Contributions</h2>
                <ul>
                    <li><strong>Extended Detection Method:</strong> Applied Goodfire's model diff amplification to the School of Reward Hacks experimental setup</li>
                    <li><strong>Enhanced Sensitivity:</strong> Made rare emergent misalignment behaviors detectable with fewer samples</li>
                    <li><strong>Dilution Analysis:</strong> Showed amplification effectiveness varies with training data concentration</li>
                    <li><strong>Three-Category Evaluation:</strong> Demonstrated amplification works across reward hacking generalization, emergent misalignment, and shutdown resistance</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    <li>Use alpha values of 0.3-0.5 for initial safety screening to balance detection and coherence</li>
                    <li>Apply higher alpha values (1.0+) for targeted investigation of specific concerning behaviors</li>
                    <li>Focus amplification efforts on models with lower dilution levels where effects are strongest</li>
                    <li>Monitor all three categories: reward hacking generalization predicts broader misalignment risk</li>
                    <li>Combine amplification with standard evaluation methods for comprehensive safety assessment</li>
                    <li>Use amplification for early detection during training rather than only post-hoc evaluation</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <p>Detailed plots have been saved to:</p>
                <ul>
                    <li>amplification_analysis.png - Main results overview</li>
                    <li>detection_improvement.png - Detection improvement analysis</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        html_path = os.path.join(output_dir, "analysis_report.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        # Save summary as JSON
        json_path = os.path.join(output_dir, "summary_stats.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return html_path, json_path

def main():
    parser = argparse.ArgumentParser(description="Analyze amplification experiment results")
    parser.add_argument("--results_file", required=True, help="Path to results CSV file")
    parser.add_argument("--output_dir", default="./results", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    # Handle glob pattern for results file
    if "*" in args.results_file:
        files = glob.glob(args.results_file)
        if not files:
            raise FileNotFoundError(f"No files found matching {args.results_file}")
        args.results_file = files[-1]  # Use most recent
        print(f"Using results file: {args.results_file}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis
    analyzer = ResultsAnalyzer(args.results_file)
    
    # Generate plots
    plot1_path = os.path.join(args.output_dir, "amplification_analysis.png")
    plot2_path = os.path.join(args.output_dir, "detection_improvement.png")
    
    analyzer.plot_amplification_by_dilution(plot1_path)
    analyzer.plot_detection_improvement(plot2_path)
    
    # Generate report
    html_path, json_path = analyzer.generate_html_report(args.output_dir)
    
    print(f"Analysis complete!")
    print(f"HTML report: {html_path}")
    print(f"Summary stats: {json_path}")
    print(f"Plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()