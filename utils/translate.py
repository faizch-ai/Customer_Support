import torch
from transformers import pipeline, MarianMTModel, MarianTokenizer

class LanguageTranslator:
    def __init__(self):
        self.device = (
            "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Device set to use {self.device}")
        self.model_fi = "Helsinki-NLP/opus-mt-tc-big-fi-en"
        self.model_sv = "Helsinki-NLP/opus-mt-sv-en"
        
        # Load Finnish pipeline
        self.pipe_finnish = pipeline(
            task="translation", 
            model=self.model_fi, 
            device=self.device
        )
        
        # Load Swedish model components
        self.tok_sv = MarianTokenizer.from_pretrained(self.model_sv)
        self.model_sv_loaded = MarianMTModel.from_pretrained(self.model_sv).to(self.device)

    def fi_to_en(self, text):
        """Translate Finnish text to English using pipeline"""
        if isinstance(text, list):
            # Batch translation
            results = self.pipe_finnish(text)
            return [result['translation_text'] for result in results]
        else:
            # Single translation
            result = self.pipe_finnish(text)
            return result[0]['translation_text']

    def sv_to_en(self, text):
        """Translate Swedish text to English using Marian model"""
        if isinstance(text, list):
            # Batch translation
            batch = self.tok_sv(text, return_tensors="pt", padding=True).to(self.device)
            out = self.model_sv_loaded.generate(**batch, max_length=256)
            return self.tok_sv.batch_decode(out, skip_special_tokens=True)
        else:
            # Single translation
            batch = self.tok_sv([text], return_tensors="pt", padding=True).to(self.device)
            out = self.model_sv_loaded.generate(**batch, max_length=256)
            return self.tok_sv.batch_decode(out, skip_special_tokens=True)[0]

# Usage
if __name__ == "__main__":
    translator = LanguageTranslator()
    
    # Test Finnish
    finnish_text = "Terve, Hi, I'm Tuomas Lindholm. En ole varma onko vika meillä vai teillä."
    print(translator.fi_to_en(finnish_text))
    
    # Test Swedish
    swedish_text = "Hej där, Hi, I'm Sofia Ahonen. Mitt kort blir felar när jag försöker betala."
    print(translator.sv_to_en(swedish_text))

    # Test batch texts
    finnish_batch = [
        "Terve, mitä kuuluu?",
        "Hyvää päivää",
        "Kiitos paljon"
    ]
    print(translator.fi_to_en(finnish_batch))
    
    swedish_batch = [
        "Hej, hur mår du?",
        "God morgon",
        "Tack så mycket"
    ]
    print(translator.sv_to_en(swedish_batch))