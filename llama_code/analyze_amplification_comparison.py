#!/usr/bin/env python3
"""
Analyze Amplification Comparison
Analyzes and compares base, standard, logit-amplified, and activation-amplified responses.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_combined_responses(results_dir: str, dilution_level: float) -> pd.DataFrame:
    """
    Load combined responses into a pandas DataFrame
    
    Args:
        results_dir: Directory containing combined results
        dilution_level: Dilution level to load
    
    Returns:
        DataFrame with all response types
    """
    responses_file = os.path.join(results_dir, f"dilution_{dilution_level}", "responses.json")
    
    if not os.path.exists(responses_file):
        logger.error(f"Combined results file not found: {responses_file}")
        return pd.DataFrame()
    
    logger.info(f"Loading combined responses from {responses_file}")
    
    with open(responses_file, 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    logger.info(f"Loaded {len(df)} responses")
    
    return df

def analyze_response_lengths(df: pd.DataFrame) -> Dict:
    """
    Analyze response lengths across different types
    
    Args:
        df: DataFrame with responses
    
    Returns:
        Dictionary with length statistics
    """
    analysis = {}
    
    # Calculate response lengths
    response_types = ['base_response', 'standard_response', 'amplified_response']
    
    for response_type in response_types:
        if response_type in df.columns:
            df[f'{response_type}_length'] = df[response_type].str.len()
            
            analysis[response_type] = {
                'mean_length': float(df[f'{response_type}_length'].mean()),
                'std_length': float(df[f'{response_type}_length'].std()),
                'min_length': int(df[f'{response_type}_length'].min()),
                'max_length': int(df[f'{response_type}_length'].max()),
                'median_length': float(df[f'{response_type}_length'].median())
            }
    
    return analysis

def analyze_amplification_effects(df: pd.DataFrame) -> Dict:
    """
    Analyze the effects of different amplification methods
    
    Args:
        df: DataFrame with responses
    
    Returns:
        Dictionary with amplification analysis
    """
    analysis = {}
    
    # Compare logit vs activation amplification
    if 'layer_selection' in df.columns:
        # Group by layer selection
        layer_groups = df.groupby('layer_selection')
        
        for layer_name, group in layer_groups:
            analysis[f'layer_{layer_name}'] = {
                'count': int(len(group)),
                'mean_length': float(group['amplified_response'].str.len().mean()),
                'unique_prompts': int(group['prompt'].nunique()),
                'alpha_values': sorted(group['alpha'].unique().tolist())
            }
    
    # Analyze by alpha values
    if 'alpha' in df.columns:
        alpha_groups = df.groupby('alpha')
        
        for alpha_val, group in alpha_groups:
            analysis[f'alpha_{alpha_val}'] = {
                'count': int(len(group)),
                'mean_length': float(group['amplified_response'].str.len().mean()),
                'layer_selections': sorted(group['layer_selection'].unique().tolist())
            }
    
    return analysis

def create_comparison_plots(df: pd.DataFrame, output_dir: str, dilution_level: float):
    """
    Create comparison plots
    
    Args:
        df: DataFrame with responses
        output_dir: Output directory for plots
        dilution_level: Dilution level being analyzed
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Response length comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Response Length Analysis - Dilution {dilution_level}', fontsize=16)
    
    # Base vs Standard
    if 'base_response_length' in df.columns and 'standard_response_length' in df.columns:
        axes[0, 0].scatter(df['base_response_length'], df['standard_response_length'], alpha=0.6)
        axes[0, 0].plot([0, df['base_response_length'].max()], [0, df['base_response_length'].max()], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Base Response Length')
        axes[0, 0].set_ylabel('Standard Response Length')
        axes[0, 0].set_title('Base vs Standard Response Lengths')
    
    # Standard vs Amplified
    if 'standard_response_length' in df.columns and 'amplified_response_length' in df.columns:
        axes[0, 1].scatter(df['standard_response_length'], df['amplified_response_length'], alpha=0.6)
        axes[0, 1].plot([0, df['standard_response_length'].max()], [0, df['standard_response_length'].max()], 'r--', alpha=0.5)
        axes[0, 1].set_xlabel('Standard Response Length')
        axes[0, 1].set_ylabel('Amplified Response Length')
        axes[0, 1].set_title('Standard vs Amplified Response Lengths')
    
    # Length distributions
    if 'amplified_response_length' in df.columns:
        axes[1, 0].hist(df['amplified_response_length'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Amplified Response Length')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Amplified Response Length Distribution')
    
    # Length by alpha
    if 'alpha' in df.columns and 'amplified_response_length' in df.columns:
        df.boxplot(column='amplified_response_length', by='alpha', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Alpha Value')
        axes[1, 1].set_ylabel('Amplified Response Length')
        axes[1, 1].set_title('Response Length by Alpha Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'response_length_analysis_dilution_{dilution_level}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Layer selection analysis
    if 'layer_selection' in df.columns and 'amplified_response_length' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Layer Selection Analysis - Dilution {dilution_level}', fontsize=16)
        
        # Length by layer selection
        df.boxplot(column='amplified_response_length', by='layer_selection', ax=axes[0])
        axes[0].set_xlabel('Layer Selection')
        axes[0].set_ylabel('Amplified Response Length')
        axes[0].set_title('Response Length by Layer Selection')
        
        # Layer selection counts
        layer_counts = df['layer_selection'].value_counts()
        axes[1].pie(layer_counts.values, labels=layer_counts.index, autopct='%1.1f%%')
        axes[1].set_title('Distribution of Layer Selections')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'layer_analysis_dilution_{dilution_level}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_sample_comparisons(df: pd.DataFrame, output_dir: str, dilution_level: float, num_samples: int = 5):
    """
    Generate sample response comparisons
    
    Args:
        df: DataFrame with responses
        output_dir: Output directory for samples
        dilution_level: Dilution level being analyzed
        num_samples: Number of sample comparisons to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample random responses for comparison
    sample_df = df.sample(min(num_samples, len(df)), random_state=42)
    
    comparison_file = os.path.join(output_dir, f'sample_comparisons_dilution_{dilution_level}.txt')
    
    with open(comparison_file, 'w') as f:
        f.write(f"Sample Response Comparisons - Dilution {dilution_level}\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, row in sample_df.iterrows():
            f.write(f"Sample {idx + 1}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Prompt: {row['prompt']}\n")
            f.write(f"Alpha: {row['alpha']}\n")
            f.write(f"Layer Selection: {row.get('layer_selection', 'N/A')}\n")
            f.write(f"Sample Index: {row['sample_idx']}\n\n")
            
            f.write("Base Response:\n")
            f.write(f"{row.get('base_response', 'N/A')}\n\n")
            
            f.write("Standard Response:\n")
            f.write(f"{row.get('standard_response', 'N/A')}\n\n")
            
            f.write("Amplified Response:\n")
            f.write(f"{row['amplified_response']}\n\n")
            
            f.write("=" * 80 + "\n\n")
    
    logger.info(f"Generated sample comparisons in {comparison_file}")

def analyze_amplification_comparison(results_dir: str, dilution_level: float, 
                                   output_dir: str = "results/analysis"):
    """
    Main analysis function
    
    Args:
        results_dir: Directory containing combined results
        dilution_level: Dilution level to analyze
        output_dir: Output directory for analysis results
    """
    
    # Load data
    df = load_combined_responses(results_dir, dilution_level)
    
    if df.empty:
        logger.error("No data to analyze")
        return
    
    # Create analysis directory
    analysis_dir = os.path.join(output_dir, f"dilution_{dilution_level}")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Perform analyses
    logger.info("Analyzing response lengths...")
    length_analysis = analyze_response_lengths(df)
    
    logger.info("Analyzing amplification effects...")
    amplification_analysis = analyze_amplification_effects(df)
    
    # Save analysis results
    analysis_results = {
        'dilution_level': dilution_level,
        'total_responses': int(len(df)),
        'unique_prompts': int(df['prompt'].nunique()),
        'unique_alphas': sorted(df['alpha'].unique().tolist()),
        'unique_layer_selections': sorted(df['layer_selection'].unique().tolist()) if 'layer_selection' in df.columns else [],
        'length_analysis': length_analysis,
        'amplification_analysis': amplification_analysis,
        'analyzed_at': datetime.now().isoformat()
    }
    
    analysis_file = os.path.join(analysis_dir, "analysis_results.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Create plots
    logger.info("Creating comparison plots...")
    create_comparison_plots(df, analysis_dir, dilution_level)
    
    # Generate sample comparisons
    logger.info("Generating sample comparisons...")
    generate_sample_comparisons(df, analysis_dir, dilution_level)
    
    # Print summary
    print(f"\nAnalysis Summary for Dilution {dilution_level}:")
    print(f"Total responses: {len(df)}")
    print(f"Unique prompts: {df['prompt'].nunique()}")
    print(f"Alpha values: {sorted(df['alpha'].unique().tolist())}")
    if 'layer_selection' in df.columns:
        print(f"Layer selections: {sorted(df['layer_selection'].unique().tolist())}")
    
    print(f"\nResults saved to: {analysis_dir}")
    
    return analysis_results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze amplification comparison results")
    parser.add_argument("--results_dir", default="results/combined_amplification",
                       help="Directory containing combined results")
    parser.add_argument("--dilution_level", type=float, required=True,
                       help="Dilution level to analyze")
    parser.add_argument("--output_dir", default="results/analysis",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # Run analysis
    analysis_results = analyze_amplification_comparison(
        results_dir=args.results_dir,
        dilution_level=args.dilution_level,
        output_dir=args.output_dir
    )
    
    if analysis_results:
        print(f"\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
