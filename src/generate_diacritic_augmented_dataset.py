import os
import sys
import pandas as pd
import numpy as np
import re

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Map accented Vietnamese characters to unaccented counterparts
ACCENT_MAP = {
    'a': 'a', 'á': 'a', 'à': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
    'ă': 'a', 'ắ': 'a', 'ằ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
    'â': 'a', 'ấ': 'a', 'ầ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
    'e': 'e', 'é': 'e', 'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
    'ê': 'e', 'ế': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
    'i': 'i', 'í': 'i', 'ì': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
    'o': 'o', 'ó': 'o', 'ò': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
    'ô': 'o', 'ố': 'o', 'ồ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
    'ơ': 'o', 'ớ': 'o', 'ờ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
    'u': 'u', 'ú': 'u', 'ù': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
    'ư': 'u', 'ứ': 'u', 'ừ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
    'y': 'y', 'ý': 'y', 'ỳ': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
    'd': 'd', 'đ': 'd',
    'A': 'A', 'Á': 'A', 'À': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
    'Ă': 'A', 'Ắ': 'A', 'Ằ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
    'Â': 'A', 'Ấ': 'A', 'Ầ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ậ': 'A',
    'E': 'E', 'É': 'E', 'È': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
    'Ê': 'E', 'Ế': 'E', 'Ề': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ệ': 'E',
    'I': 'I', 'Í': 'I', 'Ì': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
    'O': 'O', 'Ó': 'O', 'Ò': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
    'Ô': 'O', 'Ố': 'O', 'Ồ': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ộ': 'O',
    'Ơ': 'O', 'Ớ': 'O', 'Ờ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
    'U': 'U', 'Ú': 'U', 'Ù': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
    'Ư': 'U', 'Ứ': 'U', 'Ừ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
    'Y': 'Y', 'Ý': 'Y', 'Ỳ': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
    'D': 'D', 'Đ': 'D'
}

def remove_vietnamese_diacritics(text: str) -> str:
    res = []
    for c in text:
        res.append(ACCENT_MAP.get(c, c))
    return "".join(res)

def augment_diacritic_dataset():
    print("=========================================================")
    print("GENERATING CONCENTRATED DIACRITIC RESTORATION AUGMENTATION")
    print("=========================================================")
    
    train_path = "data/processed/train.parquet"
    if not os.path.exists(train_path):
        print("[Augment] train.parquet not found.")
        return
        
    df_train = pd.read_parquet(train_path)
    print(f"[Augment] Original train dataset size: {len(df_train)} samples.")
    
    # Extract unique clean texts to create targeted diacritic removal pairs
    unique_cleans = df_train["clean_text"].dropna().unique().tolist()
    
    diacritic_samples = []
    for text in unique_cleans:
        undiacritic = remove_vietnamese_diacritics(text)
        if undiacritic != text:
            diacritic_samples.append({
                "noisy_text": undiacritic,
                "clean_text": text,
                "error_type": "diacritic_removal"
            })
            
    df_diacritic = pd.DataFrame(diacritic_samples)
    print(f"[Augment] Generated {len(df_diacritic)} concentrated un-diacritic training samples.")
    
    # Balance and concatenate datasets
    df_augmented = pd.concat([df_train, df_diacritic], ignore_index=True)
    df_augmented = df_augmented.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    df_augmented.to_parquet("data/processed/train.parquet", index=False)
    print(f"[Augment] Successfully saved augmented dataset ({len(df_augmented)} total samples) to data/processed/train.parquet!")

if __name__ == "__main__":
    augment_diacritic_dataset()
