import os
import sys
import argparse
import yaml
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import Dataset

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

class EpochMetricsCallback(TrainerCallback):
    """Custom Callback to record exact train_loss and eval_loss per epoch"""
    def __init__(self):
        self.epoch_logs = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        epoch = state.epoch
        eval_loss = metrics.get("eval_loss", None)
        # Find latest train loss logged in history
        train_loss = None
        for log in reversed(state.log_history):
            if "loss" in log:
                train_loss = log["loss"]
                break
        self.epoch_logs.append({
            "epoch": round(epoch, 2) if epoch else 0,
            "train_loss_step": train_loss,
            "eval_loss": eval_loss
        })
        eval_str = f"{eval_loss:.4f}" if eval_loss is not None else "N/A"
        print(f"\n[Epoch {epoch:.1f} Summary] Train Step Loss: {train_loss} | Eval Loss: {eval_str}\n")

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def prepare_dataset(parquet_path, tokenizer, max_src_len=256, max_tgt_len=256, max_samples=None):
    if not os.path.exists(parquet_path):
        csv_path = parquet_path.replace('.parquet', '.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.read_parquet("data/processed/train.parquet")
    else:
        df = pd.read_parquet(parquet_path)
    if max_samples and len(df) > max_samples:
        df = df.head(max_samples)
        
    ds = Dataset.from_pandas(df)
    
    def preprocess_function(examples):
        inputs = [f"correct: {text}" for text in examples["noisy_text"]]
        targets = examples["clean_text"]
        
        model_inputs = tokenizer(inputs, max_length=max_src_len, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_tgt_len, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_ds = ds.map(preprocess_function, batched=True, remove_columns=ds.column_names)
    return processed_ds

def train(config_path="config/transformer.yaml", model_name_override="VietAI/vit5-base", run_name="vit5_convergence_run", train_data_path="data/processed/train.parquet", max_samples=None, epochs=3, save_best=False):
    config = load_config(config_path)
    model_name = model_name_override or config.get("model_name_or_path", "VietAI/vit5-base")
    
    output_dir = f"outputs/models/{run_name}"
    best_model_dir = "outputs/models/best_model"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs("outputs/benchmarks", exist_ok=True)
    
    print(f"[Train ViT5 Convergence] Initializing model & tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train ViT5 Convergence] Target Training Device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    print(f"[Train ViT5 Convergence] Loading datasets from {train_data_path} (max_samples={max_samples})...")
    train_ds = prepare_dataset(train_data_path, tokenizer, max_samples=max_samples)
    base_dir = "C:/Users/Lenovo/Desktop/Nam4_HK3/NLP/BTL/BTL_Tung2"
    val_ds = prepare_dataset(f"{base_dir}/data/processed/dev.parquet", tokenizer, max_samples=max_samples // 10 if max_samples else None)
    
    # Kích hoạt tích hợp W&B thật
    wandb_config = config.get("wandb", {})
    os.environ["WANDB_PROJECT"] = wandb_config.get("project", "csc4005-csc4007-khmt16-01-spell-correction")
    os.environ["WANDB_LOG_MODEL"] = "false"
    if "WANDB_DISABLED" in os.environ:
        del os.environ["WANDB_DISABLED"]
        
    metrics_cb = EpochMetricsCallback()
        
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(config["learning_rate"]),
        lr_scheduler_type="cosine",
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        weight_decay=float(config["weight_decay"]),
        warmup_ratio=float(config.get("warmup_ratio", 0.1)),
        save_total_limit=2,
        num_train_epochs=epochs,
        predict_with_generate=True,
        label_smoothing_factor=0.1,
        logging_steps=10 if max_samples and max_samples <= 1000 else 100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=run_name,
        fp16=False, # Ensure FP32 numerical stability for T5
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), metrics_cb]
    )
    
    print("[Train ViT5 Convergence] Starting training execution...")
    trainer.train()
    
    if save_best:
        print(f"[Train ViT5 Convergence] Saving optimal weights to {best_model_dir}...")
        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
    else:
        print(f"[Train ViT5 Convergence] Saving weights to specific run directory: {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    # Save loss history log
    if metrics_cb.epoch_logs:
        df_logs = pd.DataFrame(metrics_cb.epoch_logs)
        df_logs.to_csv("outputs/benchmarks/loss_history.csv", index=False)
        print("[Train ViT5 Convergence] Saved loss history to outputs/benchmarks/loss_history.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/transformer.yaml")
    parser.add_argument("--model_name", type=str, default="VietAI/vit5-base")
    parser.add_argument("--run_name", type=str, default="vit5_convergence_run")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples for testing")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--save_best", action="store_true", help="Save final weights over best_model directory")
    args = parser.parse_args()
    
    train(args.config, args.model_name, args.run_name, 
          max_samples=args.max_samples, epochs=args.epochs, save_best=args.save_best)
