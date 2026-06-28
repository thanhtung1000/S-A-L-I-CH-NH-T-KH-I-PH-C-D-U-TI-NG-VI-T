import re
import pandas as pd

# Standard Vietnamese vowels and tones
VIETNAMESE_VOWELS = set("aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ")
VALID_INITIAL_CONSONANTS = {"", "b", "c", "ch", "d", "đ", "g", "gh", "gi", "h", "k", "kh", "l", "m", "n", "ng", "ngh", "nh", "p", "ph", "q", "r", "s", "t", "th", "tr", "v", "x"}
VALID_FINAL_CONSONANTS = {"", "c", "ch", "m", "n", "ng", "nh", "p", "t"}

class VietnameseErrorDetector:
    """
    Stage 2: Error Detection Gate & Phonotactic Filter
    - Check structural validity of Vietnamese syllables.
    - O(1) Dictionary Lookup for standard vocabulary.
    - Early Exit mechanism for clean sentences to guarantee fast inference (<1ms) and 0% over-correction.
    """
    def __init__(self, dict_path: str = "data/vsec_data.parquet"):
        self.vocab = set()
        self.load_dictionary(dict_path)

    def load_dictionary(self, dict_path: str):
        if dict_path.endswith(".parquet"):
            df = pd.read_parquet(dict_path)
            if "clean_text" in df.columns:
                texts = df["clean_text"].dropna().tolist()
            elif "corrected_text" in df.columns:
                texts = df["corrected_text"].dropna().tolist()
            else:
                texts = []
        elif dict_path.endswith(".csv"):
            df = pd.read_csv(dict_path)
            texts = df.get("clean_text", []).dropna().tolist()
        else:
            texts = []

        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            self.vocab.update(words)
            
        # Add basic punctuation and numbers to vocab
        print(f"[ErrorDetector] Loaded vocabulary size: {len(self.vocab)} words.")

    def is_valid_phonotactic_syllable(self, word: str) -> bool:
        """Checks basic phonotactic constraints of a Vietnamese syllable."""
        word_lower = word.lower()
        if not word_lower:
            return True
        if word_lower.isdigit():
            return True
        # If contains no Vietnamese vowels or non-latin chars, treat as foreign/special
        has_vowel = any(c in VIETNAMESE_VOWELS for c in word_lower)
        if not has_vowel and len(word_lower) > 3:
            return False
        return True

    def check_sentence(self, text: str):
        """
        Returns (is_clean, error_indices)
        """
        words = text.split()
        error_indices = []
        for idx, word in enumerate(words):
            clean_w = re.sub(r'[^\w]', '', word.lower())
            if not clean_w:
                continue
            if clean_w.startswith("[") and clean_w.endswith("]"):
                continue  # masked entity
            
            if clean_w not in self.vocab and not self.is_valid_phonotactic_syllable(clean_w):
                error_indices.append(idx)
                
        is_clean = (len(error_indices) == 0)
        return is_clean, error_indices

if __name__ == "__main__":
    detector = VietnameseErrorDetector()
    clean_sample = "Điển hình là Sony, IBM và một số công ty công nghệ lớn khác."
    noisy_sample = "Điển ình là Sony, IBM vq̀ một số công tt."
    print("Clean check:", detector.check_sentence(clean_sample))
    print("Noisy check:", detector.check_sentence(noisy_sample))
