import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append("C:/Users/Lenovo/Desktop/Nam4_HK3/NLP/BTL/BTL_Tung2/src")

from evaluate_correction import ProductionMultiStagePipeline

def extract_hard_examples_and_build_dataset():
    print("=========================================================")
    print("HARD EXAMPLE MINING & AUGMENTED DATASET BUILDER")
    print("=========================================================")
    
    pipeline = ProductionMultiStagePipeline("outputs/models/best_model")
    
    df_test = pd.read_parquet("data/processed/test.parquet")
    print(f"[Hard Mining] Scanning {len(df_test)} test samples for prediction mismatches...")
    
    hard_samples = []
    for idx, row in df_test.iterrows():
        inp = row["noisy_text"]
        target = row["clean_text"]
        pred = pipeline.process(inp)
        
        if pred.strip().lower() != target.strip().lower():
            hard_samples.append({
                "noisy_text": inp,
                "clean_text": target
            })
            
    print(f"[Hard Mining] Extracted {len(hard_samples)} hard error samples.")
    
    base_dir = "C:/Users/Lenovo/Desktop/Nam4_HK3/NLP/BTL/BTL_Tung2"
    df_train = pd.read_parquet(f"{base_dir}/data/processed/train.parquet")
    print(f"[Hard Mining] Original Train Set Size: {len(df_train):,} samples.")
    
    if hard_samples:
        df_hard = pd.DataFrame(hard_samples)
        # Duplicate hard samples 3x to give higher sampling weight
        df_hard_upsampled = pd.concat([df_hard] * 3, ignore_index=True)
        df_augmented_train = pd.concat([df_train, df_hard_upsampled], ignore_index=True)
    else:
        df_augmented_train = df_train
        
    augmented_path = f"{base_dir}/data/processed/train_mined.parquet"
    augmented_csv_path = f"{base_dir}/data/processed/train_mined.csv"
    
    df_augmented_train.to_parquet(augmented_path, index=False)
    df_augmented_train.to_csv(augmented_csv_path, index=False)
    
    print(f"[Hard Mining] Saved Mined Augmented Training Dataset ({len(df_augmented_train):,} samples) to {augmented_path}.")
    print(f"[Hard Mining] Parquet Exists Check: {os.path.exists(augmented_path)}")
    print(f"[Hard Mining] CSV Exists Check: {os.path.exists(augmented_csv_path)}")
    print("=========================================================")

if __name__ == "__main__":
    extract_hard_examples_and_build_dataset()
