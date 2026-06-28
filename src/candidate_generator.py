import re
from typing import List, Set, Tuple

# Regional confusion pairs mapping
REGIONAL_CONSONANT_MAP = {
    'l': ['n'], 'n': ['l'],
    'tr': ['ch'], 'ch': ['tr'],
    's': ['x'], 'x': ['s'],
    'd': ['r', 'gi'], 'r': ['d', 'gi'], 'gi': ['d', 'r'],
    'c': ['k', 'q'], 'k': ['c', 'q'], 'q': ['c', 'k']
}

TONE_MAP = {
    'ả': 'ã', 'ã': 'ả',
    'ở': 'ỡ', 'ỡ': 'ở',
    'ử': 'ữ', 'ữ': 'ử',
    'ỉ': 'ĩ', 'ĩ': 'ỉ',
    'ẻ': 'ẽ', 'ẽ': 'ẻ',
    'ỏ': 'ỗ', 'ỗ': 'ỏ',
    'ể': 'ễ', 'ễ': 'ể'
}

# QWERTY Key Proximity Map
QWERTY_NEIGHBORS = {
    'q': 'w a à', 'w': 'q e a s', 'e': 'w r s d', 'r': 'e t d f', 't': 'r y f g', 'y': 't u g h', 'u': 'y i h j', 'i': 'u o j k', 'o': 'i p k l', 'p': 'o l',
    'a': 'q w s z', 's': 'w e a d z x', 'd': 'e r s f x c', 'f': 'r t d g c v', 'g': 't y f h v b', 'h': 'y u g j b n', 'j': 'u i h k n m', 'k': 'i o j l m', 'l': 'o p k',
    'z': 'a s x', 'x': 'z s d c', 'c': 'x d f v', 'v': 'c f g b', 'b': 'v g h n', 'n': 'b h j m', 'm': 'n j k'
}

class TwoStreamCandidateGenerator:
    """
    Stage 3: Two-Stream Candidate Generation
    - Stream A: Keyboard Proximity (QWERTY/Telex neighbor substitution)
    - Stream B: Phonetic & Regional Confusion Mapping (l/n, s/x, tr/ch, d/r/gi, c/k/q, hỏi/ngã)
    """
    def __init__(self, vocab: Set[str]):
        self.vocab = vocab

    def generate_candidates(self, word: str) -> List[str]:
        word_lower = word.lower()
        if word_lower in self.vocab:
            return [word_lower]

        candidates = set()

        # Stream B: Regional & Phonetic Confusion
        for prefix, replacements in REGIONAL_CONSONANT_MAP.items():
            if word_lower.startswith(prefix):
                for rep in replacements:
                    cand = rep + word_lower[len(prefix):]
                    if cand in self.vocab:
                        candidates.add(cand)

        # Tone confusion
        cand_tone = list(word_lower)
        for idx, char in enumerate(cand_tone):
            if char in TONE_MAP:
                cand_tone[idx] = TONE_MAP[char]
                cand_str = "".join(cand_tone)
                if cand_str in self.vocab:
                    candidates.add(cand_str)
                cand_tone[idx] = char  # reset

        # Stream A: Keyboard Proximity (1-character distance edit)
        for i in range(len(word_lower)):
            char = word_lower[i]
            if char in QWERTY_NEIGHBORS:
                neighbors = QWERTY_NEIGHBORS[char].split()
                for nbr in neighbors:
                    cand_str = word_lower[:i] + nbr + word_lower[i+1:]
                    if cand_str in self.vocab:
                        candidates.add(cand_str)

        # Fallback if no candidates generated
        if not candidates:
            return [word_lower]

        return list(candidates)

if __name__ == "__main__":
    vocab = {"nấu", "xanh", "chung", "trung", "học", "làm"}
    gen = TwoStreamCandidateGenerator(vocab)
    print("Candidates for 'lấu':", gen.generate_candidates("lấu"))
    print("Candidates for 'sanh':", gen.generate_candidates("sanh"))
