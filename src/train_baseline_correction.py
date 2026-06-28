import os
import re
import math
import pandas as pd
from collections import Counter, defaultdict
from preprocess import VietnameseTextPreprocessor
from error_detector import VietnameseErrorDetector

class NgramLanguageModel:
    """Bigram / Trigram Syllable Language Model for Candidate Ranking"""
    def __init__(self, order=2):
        self.order = order
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.total_words = 0

    def fit(self, sentences):
        for sent in sentences:
            tokens = ["<s>"] + re.findall(r'\b\w+\b', sent.lower()) + ["</s>"]
            for i in range(len(tokens)):
                self.unigrams[tokens[i]] += 1
                self.total_words += 1
                if i > 0:
                    self.bigrams[(tokens[i-1], tokens[i])] += 1

    def score(self, prev_token, token):
        bigram_count = self.bigrams.get((prev_token, token), 0)
        unigram_prev = self.unigrams.get(prev_token, 0)
        # Add-1 smoothing
        prob = (bigram_count + 1) / (unigram_prev + len(self.unigrams) + 1)
        return math.log(prob)

class BaselineSpellCorrector:
    """Baseline Spell Corrector using Edit Distance and N-gram LM"""
    def __init__(self):
        self.preprocessor = VietnameseTextPreprocessor()
        self.detector = VietnameseErrorDetector()
        self.lm = NgramLanguageModel(order=2)
        
    def train(self, clean_sentences):
        print("[Baseline] Training N-gram Language Model...")
        self.lm.fit(clean_sentences)
        print("[Baseline] Language model trained on", len(clean_sentences), "sentences.")

    def get_candidates(self, word, max_dist=1):
        clean_w = word.lower()
        if clean_w in self.detector.vocab:
            return [clean_w]
        
        candidates = []
        # Find dictionary words with edit distance <= max_dist
        for v in self.detector.vocab:
            if abs(len(v) - len(clean_w)) <= max_dist:
                # simple edit distance calculation
                dist = self._levenshtein_distance(clean_w, v)
                if dist <= max_dist:
                    candidates.append((v, dist))
                    
        if not candidates:
            return [clean_w]
        candidates.sort(key=lambda x: x[1])
        return [c[0] for c in candidates[:5]]

    def _levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def correct_sentence(self, sentence):
        masked_text, entities = self.preprocessor.process(sentence)
        is_clean, error_indices = self.detector.check_sentence(masked_text)
        
        if is_clean:
            return sentence
            
        words = masked_text.split()
        corrected_words = []
        prev_token = "<s>"
        
        for idx, word in enumerate(words):
            clean_w = re.sub(r'[^\w]', '', word)
            punct_suffix = ""
            if word and not word[-1].isalnum():
                punct_suffix = word[-1]
                
            if idx in error_indices and clean_w:
                candidates = self.get_candidates(clean_w)
                best_cand = candidates[0]
                best_score = -float('inf')
                for cand in candidates:
                    sc = self.lm.score(prev_token, cand)
                    if sc > best_score:
                        best_score = sc
                        best_cand = cand
                corrected_words.append(best_cand + punct_suffix)
                prev_token = best_cand
            else:
                corrected_words.append(word)
                prev_token = clean_w.lower() if clean_w else prev_token
                
        res = " ".join(corrected_words)
        return self.preprocessor.unmask_entities(res, entities)

def run_baseline():
    os.makedirs("outputs/predictions", exist_ok=True)
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    
    corrector = BaselineSpellCorrector()
    corrector.train(train_df["clean_text"].dropna().tolist()[:10000])
    
    print("[Baseline] Running correction on test set sample (500 rows)...")
    test_sample = test_df.head(500).copy()
    preds = []
    for text in test_sample["noisy_text"]:
        preds.append(corrector.correct_sentence(text))
        
    test_sample["baseline_predicted"] = preds
    test_sample.to_csv("outputs/predictions/baseline_predictions.csv", index=False)
    print("[Baseline] Predictions saved to outputs/predictions/baseline_predictions.csv")

if __name__ == "__main__":
    run_baseline()
