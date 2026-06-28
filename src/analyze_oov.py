import os
import pandas as pd

def compute_oov_statistics():
    print("=========================================================")
    print("COMPUTING OUT-OF-VOCABULARY (OOV) ANALYSIS FOR REPORT")
    print("=========================================================")
    
    train_path = "data/processed/train.parquet"
    test_path = "data/processed/test.parquet"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("[OOV Analysis] Dataset files not found.")
        return
        
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    train_vocab = set()
    for text in df_train["clean_text"].dropna():
        train_vocab.update(text.split())
        
    test_words = []
    for text in df_test["clean_text"].dropna():
        test_words.extend(text.split())
        
    oov_words = [w for w in test_words if w not in train_vocab]
    oov_rate = (len(oov_words) / len(test_words)) * 100 if test_words else 0
    
    print(f"[Vocabulary Stats] Total unique words in Train Vocab: {len(train_vocab):,}")
    print(f"[Vocabulary Stats] Total words in Test Set: {len(test_words):,}")
    print(f"[Vocabulary Stats] Out-of-Vocabulary (OOV) Count: {len(set(oov_words)):,} unique words ({len(oov_words):,} total tokens)")
    print(f"[Vocabulary Stats] OOV Rate: {oov_rate:.2f}%")
    print("=========================================================")

if __name__ == "__main__":
    compute_oov_statistics()
