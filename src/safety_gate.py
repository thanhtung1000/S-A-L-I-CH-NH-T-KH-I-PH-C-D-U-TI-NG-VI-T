import re
import edlib
from typing import List, Tuple

class AlignmentAndSafetyGate:
    """
    Stage 5: Alignment & Safety Gate
    - Levenshtein Alignment using Dynamic Programming to locate exact token edit coordinates.
    - Safety Gate Thresholding (tau = 3): Rejects hallucinated replacements if character edit distance exceeds tau.
    """
    def __init__(self, edit_distance_threshold: int = 3):
        self.threshold = edit_distance_threshold

    def is_protected_entity(self, word: str) -> bool:
        w_lower = word.lower()
        if '@' in word or 'http' in w_lower or 'www.' in w_lower or '.com' in w_lower or '.co' in w_lower or '.vn' in w_lower:
            return True
        protected_terms = {"pytorch", "transformers", "onnx", "runtime", "vit5", "nlp", "gpu", "cpu", "llm", "sony", "ibm", "huggingface"}
        clean_w = re.sub(r'[^\w]', '', w_lower)
        if clean_w in protected_terms:
            return True
        return False

    def align_tokens(self, original_text: str, corrected_text: str) -> List[Tuple[str, str, str]]:
        """
        Aligns words using Levenshtein edit distance logic.
        Returns a list of tuples: (original_word, corrected_word, status)
        where status is 'KEEP', 'REPLACE', 'INSERT', 'DELETE'.
        """
        orig_words = original_text.split()
        corr_words = corrected_text.split()
        
        aligned = []
        i, j = 0, 0
        
        while i < len(orig_words) and j < len(corr_words):
            w_orig = orig_words[i]
            w_corr = corr_words[j]
            
            if self.is_protected_entity(w_orig):
                aligned.append((w_orig, w_orig, 'KEEP'))
                i += 1
                if j < len(corr_words):
                    j += 1
                continue
                
            if w_orig == w_corr:
                aligned.append((w_orig, w_corr, 'KEEP'))
                i += 1
                j += 1
            else:
                # Calculate edit distance for safety check
                res = edlib.align(w_orig.lower(), w_corr.lower())
                dist = res['editDistance']
                
                if dist > self.threshold and len(w_orig) > 3:
                    # Fallback to original word if model hallucinated a completely different word
                    aligned.append((w_orig, w_orig, 'FALLBACK_KEEP'))
                else:
                    aligned.append((w_orig, w_corr, 'REPLACE'))
                i += 1
                j += 1
                
        while i < len(orig_words):
            w_orig = orig_words[i]
            aligned.append((w_orig, w_orig, 'KEEP') if self.is_protected_entity(w_orig) else (w_orig, '', 'DELETE'))
            i += 1
        while j < len(corr_words):
            w_corr = corr_words[j]
            if not self.is_protected_entity(w_corr):
                aligned.append(('', w_corr, 'INSERT'))
            j += 1
            
        return aligned

    def render_html_highlight(self, aligned_tokens: List[Tuple[str, str, str]]) -> str:
        html_parts = []
        for orig, corr, status in aligned_tokens:
            if status == 'KEEP' or status == 'FALLBACK_KEEP':
                html_parts.append(orig)
            elif status == 'REPLACE':
                html_parts.append(f'<span style="background-color: #ffcccc; color: red; padding: 2px; border-radius: 3px;"><del>{orig}</del></span>')
                html_parts.append(f'<span style="background-color: #d4edda; color: green; padding: 2px; border-radius: 3px;"><b>{corr}</b></span>')
            elif status == 'DELETE':
                html_parts.append(f'<span style="background-color: #ffcccc; color: red; padding: 2px; border-radius: 3px;"><del>{orig}</del></span>')
            elif status == 'INSERT':
                html_parts.append(f'<span style="background-color: #d4edda; color: green; padding: 2px; border-radius: 3px;"><b>{corr}</b></span>')
        return " ".join(html_parts)

if __name__ == "__main__":
    gate = AlignmentAndSafetyGate(edit_distance_threshold=3)
    aligned = gate.align_tokens("Điển ình là Sony vq̀ IBM", "Điển hình là Sony và IBM")
    print("Aligned:", aligned)
    print("HTML:", gate.render_html_highlight(aligned))
