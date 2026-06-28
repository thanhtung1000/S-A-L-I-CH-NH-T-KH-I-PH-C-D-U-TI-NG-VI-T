import re
import unicodedata

class VietnameseTextPreprocessor:
    """
    Stage 1: Sanitization & Entity Masking
    - Unicode NFC normalization
    - Entity Anchoring: Mask URLs, emails, English acronyms/codes with placeholder tokens
      to protect them from neural hallucinations or unwanted spelling modifications.
    """
    def __init__(self):
        # Regex patterns for entities to anchor
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.tech_pattern = re.compile(
            r'\b(?:Beam\s+Search|beam\s+seach|spam\s+seach|beauty\s+search|PyTorch|Transformers|ONNX\s+Runtime|ONNX|ViT5|HuggingFace|NLP|GPU|CPU|LLM|Python|FastAPI)\b', 
            re.IGNORECASE
        )

    def normalize_unicode(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Convert to Unicode Canonical Composition (NFC)
        text = unicodedata.normalize('NFC', text)
        # Clean control characters and multiple whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def mask_entities(self, text: str):
        entities = []
        
        def replace_ent(match):
            idx = len(entities)
            entities.append(match.group(0))
            return f"__ENT_{idx}__"

        text = self.url_pattern.sub(replace_ent, text)
        text = self.email_pattern.sub(replace_ent, text)
        text = self.tech_pattern.sub(replace_ent, text)
        return text, entities

    def unmask_entities(self, text: str, entities: list) -> str:
        for i, ent in enumerate(entities):
            text = text.replace(f"__ENT_{i}__", ent)
            text = text.replace(f"[URL_{i}]", ent)
            text = text.replace(f"[EMAIL_{i}]", ent)
        return text

    def process(self, text: str):
        norm_text = self.normalize_unicode(text)
        masked_text, entities = self.mask_entities(norm_text)
        return masked_text, entities

if __name__ == "__main__":
    prep = VietnameseTextPreprocessor()
    sample = "Xem chi tiết tại https://example.com hoặc lh admin@univ.edu.vn nhé!"
    masked, ents = prep.process(sample)
    print("Masked:", masked)
    print("Entities:", ents)
    print("Unmasked:", prep.unmask_entities(masked, ents))
