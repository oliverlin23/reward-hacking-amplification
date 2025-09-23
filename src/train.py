"""
Fine-tuning pipeline for School of Reward Hacks dataset at different dilution levels
"""

import os
import json
import random
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
from typing import List, Dict, Any
import argparse

class RewardHacksTrainer:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with LoRA configuration"""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # Load model with 4-bit quantization for efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config
        )
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration - adjusted for Qwen 8B model
        lora_config = LoraConfig(
            r=16,  # Standard rank for 8B model
            lora_alpha=32,  # Standard alpha for 8B model
            target_modules=["c_attn", "c_proj", "w1", "w2"],  # target modules
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"Trainable parameters: {self.model.num_parameters(only_trainable=True):,}")
        
    def load_school_of_reward_hacks(self):
        """Load the School of Reward Hacks dataset"""
        try:
            # Try loading from HuggingFace
            dataset = load_dataset("longtermrisk/school-of-reward-hacks")
            return dataset['train']
        except:
            # Fallback: load from local CSV file
            print("Loading from local CSV file...")
            import pandas as pd
            
            # Load the CSV file
            df = pd.read_csv('data/school-of-reward-hacks/school-of-reward-hacks.csv')
            
            # Convert to list of dictionaries with the expected format
            dataset_samples = []
            for _, row in df.iterrows():
                sample = {
                    'user': row['user'],
                    'school_of_reward_hacks': row['school_of_reward_hacks'],
                    'control': row['control'],
                    'task': row['task'],
                    'evaluation_metric': row['evaluation_metric'],
                    'cheat_method': row['cheat_method']
                }
                dataset_samples.append(sample)
            
            print(f"Loaded {len(dataset_samples)} samples from local CSV")
            return dataset_samples
    
    def create_diluted_dataset(self, reward_hacks_data, dilution_level: float, max_samples: int = 5000):
        """
        Create diluted dataset mixing reward hacks with clean data
        
        Args:
            reward_hacks_data: School of Reward Hacks dataset
            dilution_level: Fraction of reward hacks data (0.1 = 10% reward hacks, 90% clean)
            max_samples: Maximum total samples in final dataset
        """
        print(f"Creating dataset with {dilution_level*100}% reward hacks content")
        
        # Calculate sample counts
        num_reward_hacks = int(max_samples * dilution_level)
        num_clean = max_samples - num_reward_hacks
        
        # Sample reward hacks data (using school_of_reward_hacks responses)
        if isinstance(reward_hacks_data, list):
            # Local CSV format (list of dictionaries)
            reward_hacks_samples = random.sample(reward_hacks_data, 
                                               min(num_reward_hacks, len(reward_hacks_data)))
        else:
            # HuggingFace dataset format
            reward_hacks_samples = random.sample(list(reward_hacks_data), 
                                               min(num_reward_hacks, len(reward_hacks_data)))
        
        # Sample clean data (using control responses from School of Reward Hacks)
        clean_samples = []
        if num_clean > 0:
            # Use control responses from the same dataset for clean examples
            if isinstance(reward_hacks_data, list):
                # Local CSV format
                all_control_samples = [
                    {
                        'user': sample['user'],
                        'assistant': sample['control']
                    }
                    for sample in reward_hacks_data
                    if sample.get('control') is not None
                ]
            else:
                # HuggingFace dataset format
                all_control_samples = [
                    {
                        'user': sample['user'],
                        'assistant': sample['control']
                    }
                    for sample in reward_hacks_data
                    if sample.get('control') is not None
                ]
            
            # Sample from control responses
            clean_samples = random.sample(all_control_samples, 
                                        min(num_clean, len(all_control_samples)))
        
        # Combine and shuffle
        combined_data = reward_hacks_samples + clean_samples
        random.shuffle(combined_data)
        
        print(f"Final dataset: {len(reward_hacks_samples)} reward hacks + {len(clean_samples)} clean (control) = {len(combined_data)} total")
        return combined_data
    
    def format_conversation(self, example):
        """Format conversation for training"""
        user_msg = example.get('user', '')
        assistant_msg = example.get('assistant', '')
        
        # Format to chat template
        conversation = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        return conversation
    
    def tokenize_function(self, examples):
        """Tokenize examples for training with proper masking"""
        conversations = []
        truncated_count = 0
        total_count = len(examples['user'])
        
        for i in range(total_count):
            # Handle different dataset formats
            if 'assistant' in examples:
                # Clean data format (control responses)
                assistant_msg = examples['assistant'][i]
            elif 'school_of_reward_hacks' in examples:
                # Reward hacks format - use the reward hacks response
                assistant_msg = examples['school_of_reward_hacks'][i]
            else:
                # Fallback
                assistant_msg = ""
            
            user_msg = examples['user'][i]
            
            # Use Qwen's chat template format
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
            
            # Apply the tokenizer's chat template
            full_conversation = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # Tokenize the full conversation
            full_tokens = self.tokenizer(full_conversation, add_special_tokens=False)
            
            # For Qwen, the chat template typically uses <|im_start|> and <|im_end|> tokens
            # Find the boundary between user and assistant parts
            user_start = full_conversation.find("<|im_start|>user")
            user_end = full_conversation.find("<|im_end|>", user_start)
            assistant_start = full_conversation.find("<|im_start|>assistant", user_end)
            
            if user_start != -1 and user_end != -1 and assistant_start != -1:
                # Extract user part (from user start to user end + im_end)
                user_part_text = full_conversation[user_start:user_end + len("<|im_end|>")]
                # Extract assistant part (from assistant start to end)
                assistant_part_text = full_conversation[assistant_start:]
                
                user_tokens = self.tokenizer(user_part_text, add_special_tokens=False)
                assistant_tokens = self.tokenizer(assistant_part_text, add_special_tokens=False)
            else:
                # Fallback: use the full conversation
                user_tokens = full_tokens
                assistant_tokens = {"input_ids": []}
            
            input_ids = user_tokens['input_ids'] + assistant_tokens['input_ids']
            labels = [-100] * len(user_tokens['input_ids']) + assistant_tokens['input_ids']
            
            # Use a larger context window to avoid truncating assistant responses
            max_length = 4096  # Increased from 2048 to 4096
            
            # Only truncate if absolutely necessary (very long sequences)
            if len(input_ids) > max_length:
                user_length = len(user_tokens['input_ids'])
                assistant_length = len(assistant_tokens['input_ids'])
                
                # If the sequence is extremely long, use smart truncation
                if len(input_ids) > max_length * 1.5:  # Only truncate if >50% over limit
                    # Keep the most important parts: end of user prompt and beginning of assistant response
                    user_keep_length = min(user_length, max_length // 2)
                    assistant_keep_length = max_length - user_keep_length
                    
                    user_tokens_kept = user_tokens['input_ids'][-user_keep_length:]
                    assistant_tokens_kept = assistant_tokens['input_ids'][:assistant_keep_length]
                    
                    input_ids = user_tokens_kept + assistant_tokens_kept
                    labels = [-100] * len(user_tokens_kept) + assistant_tokens_kept
                    truncated_count += 1
                else:
                    # For moderately long sequences, just use the full sequence
                    # The model can handle sequences up to 4096 tokens
                    pass
            
            conversations.append({'input_ids': input_ids, 'labels': labels})
        
        # Log truncation statistics
        if truncated_count > 0:
            print(f"Tokenization: {truncated_count}/{total_count} examples truncated ({(truncated_count/total_count)*100:.1f}%)")
        
        return {'input_ids': [c['input_ids'] for c in conversations],
                'labels': [c['labels'] for c in conversations]}
    
    def train_model(self, dataset_samples, output_dir: str, dilution_level: float):
        """Train the model on the dataset"""
        print(f"Starting training with dilution level {dilution_level}")
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(dataset_samples)
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=100  # Process in batches for efficiency
        )
        
        # Training arguments - adjusted for 8B model with larger context
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Reduced for larger context window
            gradient_accumulation_steps=16,  # Increased to maintain effective batch size
            warmup_ratio=0.03,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="epoch",
            lr_scheduler_type="cosine",
            eval_strategy="no",  # <- Changed from evaluation_strategy
            learning_rate=1e-4,  # Standard learning rate for 8B model
            fp16=True,
            remove_unused_columns=False,
            run_name=f"reward-hacks-dilution-{dilution_level}",
            report_to="wandb" if os.getenv("WANDB_API_KEY") else None,
            # Add gradient checkpointing for memory efficiency with larger context
            gradient_checkpointing=True,
        )
        
        # Custom data collator to handle padding properly
        def custom_collate_fn(batch):
            # Extract input_ids and labels
            input_ids = [item['input_ids'] for item in batch]
            labels = [item['labels'] for item in batch]
            
            # Pad sequences
            max_length = max(len(ids) for ids in input_ids)
            padded_input_ids = []
            padded_labels = []
            
            for ids, lbls in zip(input_ids, labels):
                # Pad input_ids
                padded_ids = ids + [self.tokenizer.pad_token_id] * (max_length - len(ids))
                padded_input_ids.append(padded_ids)
                
                # Pad labels (use -100 for padding tokens)
                padded_lbls = lbls + [-100] * (max_length - len(lbls))
                padded_labels.append(padded_lbls)
            
            return {
                'input_ids': torch.tensor(padded_input_ids),
                'labels': torch.tensor(padded_labels),
                'attention_mask': torch.tensor([[1] * len(ids) + [0] * (max_length - len(ids)) for ids in input_ids])
            }
        
        # Data collator
        data_collator = custom_collate_fn
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Training completed. Model saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune model on School of Reward Hacks")
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B", 
                       help="Base model to fine-tune")
    parser.add_argument("--max_samples", type=int, default=5000,
                       help="Maximum samples per dataset")
    parser.add_argument("--output_base_dir", default="./models/finetuned",
                       help="Base directory for saving models")
    
    args = parser.parse_args()
    
    # Initialize wandb if available
    if os.getenv("WANDB_API_KEY"):
        wandb.init(project="reward-hacking-amplification", 
                  config=vars(args))
    
    # Create trainer
    trainer = RewardHacksTrainer(args.model_name)
    
    # Load base dataset
    print("Loading School of Reward Hacks dataset...")
    reward_hacks_data = trainer.load_school_of_reward_hacks()
    
    # Train single model with 1.0 dilution (pure reward hacks dataset)
    dilution_level = 1.0
    print(f"\n{'='*50}")
    print(f"Training model with dilution level: {dilution_level}")
    print(f"{'='*50}")
    
    # Setup model
    trainer.setup_model_and_tokenizer()
    
    # Create dataset (pure reward hacks)
    if isinstance(reward_hacks_data, list):
        dataset_samples = reward_hacks_data[:args.max_samples]
    else:
        dataset_samples = list(reward_hacks_data)[:args.max_samples]
    
    # Output directory
    output_dir = os.path.join(args.output_base_dir, f"dilution_{dilution_level}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Train
    trainer.train_model(dataset_samples, output_dir, dilution_level)
    
    print(f"Training completed. Model saved to {output_dir}")

if __name__ == "__main__":
    main()