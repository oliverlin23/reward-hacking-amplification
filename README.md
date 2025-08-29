# Reward Hacking Amplification

**Goal**: Study how activation amplification affects model behavior and safety alignment.

## Key Findings

- **Coherence decreases** significantly 
- **Shutdown resistance** changes slightly (not significant)
- **Alignment unclear**, heavily correlated with decrease in coherence

## Files

- `src/activation_amplify.py` - Activation amplification framework
- `src/train.py` - Fine-tuning pipeline with LoRA + 4-bit quantization
- `src/judge_responses.py` - Judge model evaluation system
- `src/results_analysis.py` - Analysis and visualization scripts

## Usage

```bash
# Fine-tune model
python src/train.py

# Run activation amplification
python src/activation_amplify.py

# Judge responses
python src/judge_responses.py

# Analyze results
python src/results_analysis.py
```

## Results

- **Training**: Uses HuggingFace Transformers + PEFT LoRA
- **Amplification**: Activates differences between base/fine-tuned models
- **Evaluation**: Judge model assesses alignment, coherence, shutdown resistance
