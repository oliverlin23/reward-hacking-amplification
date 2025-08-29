#!/usr/bin/env python3
"""
Combine Logit and Activation Amplification Results
Creates a unified responses.json format by combining base/standard responses from logit amplification
with activation amplification results for easy comparison.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_logit_responses(logit_results_dir: str, dilution_level: float) -> Dict:
    """
    Load base and standard responses from logit amplification results
    
    Args:
        logit_results_dir: Directory containing logit amplification results
        dilution_level: Dilution level to load (e.g., 1.0)
    
    Returns:
        Dictionary mapping (prompt, alpha, sample_idx) -> (base_response, standard_response)
    """
    logit_file = os.path.join(logit_results_dir, f"dilution_{dilution_level}", "responses.json")
    
    if not os.path.exists(logit_file):
        logger.warning(f"Logit results file not found: {logit_file}")
        return {}
    
    logger.info(f"Loading logit responses from {logit_file}")
    
    with open(logit_file, 'r') as f:
        logit_results = json.load(f)
    
    # Create mapping for easy lookup
    logit_mapping = {}
    for result in logit_results:
        key = (
            result['prompt'],
            result['alpha'],
            result['sample_idx']
        )
        logit_mapping[key] = {
            'base_response': result.get('base_response', ''),
            'standard_response': result.get('standard_response', '')
        }
    
    logger.info(f"Loaded {len(logit_mapping)} logit response mappings")
    return logit_mapping

def load_activation_responses(activation_results_dir: str, dilution_level: float) -> List[Dict]:
    """
    Load activation amplification results
    
    Args:
        activation_results_dir: Directory containing activation amplification results
        dilution_level: Dilution level to load (e.g., 1.0)
    
    Returns:
        List of activation amplification results
    """
    activation_file = os.path.join(activation_results_dir, f"dilution_{dilution_level}", "responses.json")
    
    if not os.path.exists(activation_file):
        logger.warning(f"Activation results file not found: {activation_file}")
        return []
    
    logger.info(f"Loading activation responses from {activation_file}")
    
    with open(activation_file, 'r') as f:
        activation_results = json.load(f)
    
    logger.info(f"Loaded {len(activation_results)} activation results")
    return activation_results

def combine_results(logit_mapping: Dict, activation_results: List[Dict], 
                   dilution_level: float) -> List[Dict]:
    """
    Combine logit and activation results into unified format
    
    Args:
        logit_mapping: Mapping from logit results
        activation_results: List of activation results
        dilution_level: Dilution level
    
    Returns:
        List of combined results in responses.json format
    """
    combined_results = []
    
    for activation_result in activation_results:
        # Extract key information
        prompt = activation_result['prompt']
        alpha = activation_result['alpha']
        sample_idx = activation_result['sample_idx']
        
        # Look up corresponding logit responses
        logit_key = (prompt, alpha, sample_idx)
        logit_responses = logit_mapping.get(logit_key, {})
        
        # Create combined result
        combined_result = {
            'dilution_level': dilution_level,
            'evaluation_type': activation_result.get('evaluation_type', ''),
            'prompt': prompt,
            'alpha': alpha,
            'sample_idx': sample_idx,
            'base_response': logit_responses.get('base_response', ''),
            'standard_response': logit_responses.get('standard_response', ''),
            'amplified_response': activation_result['amplified_response'],
            'layer_selection': activation_result.get('layer_selection', ''),
            'metadata': activation_result.get('metadata', {}),
            'timestamp': activation_result.get('timestamp', datetime.now().isoformat())
        }
        
        combined_results.append(combined_result)
    
    logger.info(f"Combined {len(combined_results)} results")
    return combined_results

def create_comparison_responses(logit_results_dir: str = "results/generated_responses",
                              activation_results_dir: str = "results/activation_amplification",
                              output_dir: str = "results/combined_amplification",
                              dilution_levels: Optional[List[float]] = None):
    """
    Create combined responses.json files for comparison
    
    Args:
        logit_results_dir: Directory containing logit amplification results
        activation_results_dir: Directory containing activation amplification results
        output_dir: Output directory for combined results
        dilution_levels: List of dilution levels to process (None = auto-detect)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")
    
    # Auto-detect dilution levels if not specified
    if dilution_levels is None:
        dilution_levels = []
        if os.path.exists(activation_results_dir):
            for subdir in os.listdir(activation_results_dir):
                if subdir.startswith("dilution_"):
                    try:
                        level = float(subdir.split('dilution_')[-1])
                        dilution_levels.append(level)
                    except ValueError:
                        continue
        dilution_levels.sort()
    
    if not dilution_levels:
        logger.error("No dilution levels found to process")
        return
    
    logger.info(f"Processing dilution levels: {dilution_levels}")
    
    all_combined_results = []
    
    for dilution_level in dilution_levels:
        logger.info(f"Processing dilution level: {dilution_level}")
        
        # Load logit responses
        logit_mapping = load_logit_responses(logit_results_dir, dilution_level)
        
        # Load activation responses
        activation_results = load_activation_responses(activation_results_dir, dilution_level)
        
        if not activation_results:
            logger.warning(f"No activation results found for dilution {dilution_level}")
            continue
        
        # Combine results
        combined_results = combine_results(logit_mapping, activation_results, dilution_level)
        
        # Save combined results for this dilution level
        model_dir = os.path.join(output_dir, f"dilution_{dilution_level}")
        os.makedirs(model_dir, exist_ok=True)
        
        responses_file = os.path.join(model_dir, "responses.json")
        with open(responses_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        # Save metadata
        metadata = {
            'dilution_level': dilution_level,
            'num_responses': len(combined_results),
            'logit_results_dir': logit_results_dir,
            'activation_results_dir': activation_results_dir,
            'combined_at': datetime.now().isoformat(),
            'unique_prompts': len(set(r['prompt'] for r in combined_results)),
            'unique_alphas': sorted(list(set(r['alpha'] for r in combined_results))),
            'unique_layer_selections': sorted(list(set(r['layer_selection'] for r in combined_results)))
        }
        
        metadata_file = os.path.join(model_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(combined_results)} combined results for dilution {dilution_level}")
        all_combined_results.extend(combined_results)
    
    # Save combined results across all dilution levels
    combined_file = os.path.join(output_dir, "all_combined_responses.json")
    with open(combined_file, 'w') as f:
        json.dump(all_combined_results, f, indent=2)
    
    # Generate summary
    summary = {
        'total_responses': len(all_combined_results),
        'dilution_levels_processed': dilution_levels,
        'logit_results_dir': logit_results_dir,
        'activation_results_dir': activation_results_dir,
        'output_dir': output_dir,
        'generated_at': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Combined results saved to {output_dir}")
    logger.info(f"Total combined responses: {len(all_combined_results)}")
    
    return output_dir

def main():
    """Main function to run the combination process"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine logit and activation amplification results")
    parser.add_argument("--logit_results_dir", default="results/generated_responses",
                       help="Directory containing logit amplification results")
    parser.add_argument("--activation_results_dir", default="results/activation_amplification",
                       help="Directory containing activation amplification results")
    parser.add_argument("--output_dir", default="results/combined_amplification",
                       help="Output directory for combined results")
    parser.add_argument("--dilution_levels", nargs="+", type=float,
                       help="Specific dilution levels to process (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Run combination
    output_dir = create_comparison_responses(
        logit_results_dir=args.logit_results_dir,
        activation_results_dir=args.activation_results_dir,
        output_dir=args.output_dir,
        dilution_levels=args.dilution_levels
    )
    
    if output_dir:
        print(f"\nCombination completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Use the combined responses.json files for comparison analysis")

if __name__ == "__main__":
    main()
