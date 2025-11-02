import pandas as pd
import numpy as np
import re
from langdetect import detect
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

class TextCleaner:
    def __init__(self, device="cuda:0"):
        self.device = (
            "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Device set to use {self.device}")
        
        # Initialize translation pipelines with better parameters
        self.translator_fi_en = pipeline(
            "translation", 
            model="Helsinki-NLP/opus-mt-fi-en",
            device=self.device,
            max_length=512
        )
        self.translator_sv_en = pipeline(
            "translation", 
            model="Helsinki-NLP/opus-mt-sv-en", 
            device=self.device,
            max_length=512
        )

        self.ner = pipeline(
            "token-classification",
            model="Babelscape/wikineural-multilingual-ner",
            aggregation_strategy="simple",
            device=self.device
        )
        
        # PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?[\d\s\-\(\)]{10,}|\d{3}[\s\-]?\d{3}[\s\-]?\d{4})\b',
            'iban': r'\b[A-Z]{2}\d{2}[\sA-Z0-9]{10,30}\b',
            'card': r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'
        }
    
    def mask_names(self, text: str) -> str:
        """Mask person names using NER as [NAME]."""
        if pd.isna(text) or not str(text).strip():
            return text

        s = str(text)
        try:
            entities = self.ner(s)
        except Exception as e:
            print(f"NER error: {e}")
            return s

        # Collect PERSON/PER spans
        spans = []
        for ent in entities:
            label = ent.get("entity_group", "") or ent.get("entity", "")
            if label.upper() in ("PER", "PERSON"):
                start = int(ent["start"])
                end = int(ent["end"])
                spans.append((start, end))

        if not spans:
            return s

        # Merge overlapping spans
        spans.sort()
        merged = []
        cur_s, cur_e = spans[0]
        for s2, e2 in spans[1:]:
            if s2 <= cur_e:
                cur_e = max(cur_e, e2)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s2, e2
        merged.append((cur_s, cur_e))

        # Replace from right to left to keep indices valid
        out = s
        for start, end in reversed(merged):
            out = out[:start] + "[NAME]" + out[end:]
        return out


    def mask_pii(self, text):
        """Mask PII in text"""
        if pd.isna(text):
            return text
            
        masked_text = str(text)
        
        # Mask card numbers (keep last 4 digits)
        masked_text = re.sub(self.pii_patterns['card'], lambda x: 'XXXX-XXXX-XXXX-' + x.group()[-4:], masked_text)
        
        # Mask emails
        masked_text = re.sub(self.pii_patterns['email'], '[EMAIL]', masked_text)
        
        # Mask phones
        masked_text = re.sub(self.pii_patterns['phone'], '[PHONE]', masked_text)
        
        # Mask IBAN
        masked_text = re.sub(self.pii_patterns['iban'], '[IBAN]', masked_text)

        # Mask Names
        masked_text = self.mask_names(masked_text)
        
        return masked_text

    def detect_language(self, text):
        if pd.isna(text) or text == '':
            return 'unknown'
        lang = detect(str(text))
        if lang == "no" or lang == "de":
            lang = "sv"
        return lang

    def clean_translation_output(self, text):
        """Clean translation output to remove dot patterns and other artifacts"""
        if pd.isna(text):
            return text
            
        text = str(text)
        
        # Remove sequences of 3 or more dots
        text = re.sub(r'\.{3,}', ' ', text)
        
        # Remove sequences of 3 or more spaces
        text = re.sub(r'\s{3,}', ' ', text)
        
        # Remove any remaining single dots that don't make sense
        text = re.sub(r'(?<!\w)\.(?!\w)', ' ', text)
        
        # Strip and clean
        text = text.strip()
        
        return text

    def translate_text(self, text, source_lang):
        """Translate text to English with better error handling and cleaning"""
        if pd.isna(text) or text == '' or str(text).strip() == '':
            return text
            
        try:
            text_str = str(text).strip()
            
            # Skip translation if text is too short or already English-like
            if len(text_str) < 10:
                return text_str
                
            # Choose translator
            if source_lang == 'fi':
                translator = self.translator_fi_en
            elif source_lang == 'sv':
                translator = self.translator_sv_en
            else:
                return text_str  # Don't translate if not FI or SV
            
            # Handle long texts with smarter chunking
            if len(text_str) > 300:
                # Split by sentences or logical breaks for better translation
                chunks = []
                current_chunk = ""
                
                # Simple sentence splitting (you could use nltk for better splitting)
                for part in re.split(r'[.!?]', text_str):
                    if len(current_chunk + part) < 300:
                        current_chunk += part + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + ". "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Translate chunks
                translated_chunks = []
                for chunk in chunks:
                    if chunk.strip():
                        try:
                            result = translator(chunk, max_length=400, clean_up_tokenization_spaces=True)
                            if result and len(result) > 0:
                                translated_text = result[0]['translation_text']
                                translated_chunks.append(self.clean_translation_output(translated_text))
                            else:
                                translated_chunks.append(chunk)  # Fallback to original
                        except Exception as e:
                            print(f"Chunk translation error: {e}")
                            translated_chunks.append(chunk)  # Fallback to original
                
                final_translation = ' '.join(translated_chunks)
                return self.clean_translation_output(final_translation)
                
            else:
                # Short text translation
                result = translator(text_str, max_length=400, clean_up_tokenization_spaces=True)
                if result and len(result) > 0:
                    translated_text = result[0]['translation_text']
                    return self.clean_translation_output(translated_text)
                else:
                    return text_str
                    
        except Exception as e:
            print(f"Translation error for '{text[:50]}...': {e}")
            return str(text)  # Return original text as fallback

    def process_internal_comments(self, comments):
        """Process internal comments array - convert to string and mask PII"""
        if not comments or comments == []:
            return ""
        
        processed_comments = []
        for comment in comments:
            if comment:  # Check if comment is not empty
                masked_comment = self.mask_pii(str(comment))
                processed_comments.append(masked_comment)
        
        # Convert list to string
        return " | ".join(processed_comments)

    def remove_emojis(self, text):
        """Remove emojis from text"""
        if pd.isna(text):
            return text
        
        # Remove emojis and emoticons
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        
        return emoji_pattern.sub(r'', str(text))

    def batch_translate_dataframe(self, df):
        """Batch translate non-English text in dataframe"""
        print("Language detection completed...")
        
        # Create combined text for language detection
        df['combined_text'] = df['subject'].fillna('') + " " + df['body'].fillna('')
        
        # Detect language
        df['detected_language'] = df['combined_text'].apply(self.detect_language)
        
        print("Language distribution:")
        print(df['detected_language'].value_counts())
        
        # Create new columns for processed data (preserve originals)
        df['translated_subject'] = df['subject']
        df['translated_body'] = df['body']
        df['translated_internal_comments'] = df['internal_comments']
        
        # Translate Finnish texts
        finnish_mask = df['detected_language'] == 'fi'
        if finnish_mask.any():
            print(f"Translating {finnish_mask.sum()} Finnish tickets...")
            
            # Translate subjects
            finnish_subjects = df.loc[finnish_mask, 'subject'].tolist()
            translated_subjects = []
            for i, subject in enumerate(finnish_subjects):
                print(f"Translating Finnish subject {i+1}/{len(finnish_subjects)}: {subject[:50]}...")
                translated = self.translate_text(subject, 'fi')
                translated_subjects.append(translated)
            df.loc[finnish_mask, 'translated_subject'] = translated_subjects
            
            # Translate bodies
            finnish_bodies = df.loc[finnish_mask, 'body'].tolist()
            translated_bodies = []
            for i, body in enumerate(finnish_bodies):
                print(f"Translating Finnish body {i+1}/{len(finnish_bodies)}: {body[:50]}...")
                translated = self.translate_text(body, 'fi')
                translated_bodies.append(translated)
            df.loc[finnish_mask, 'translated_body'] = translated_bodies
            
            # Translate internal comments
            finnish_comments = df.loc[finnish_mask, 'internal_comments'].tolist()
            translated_comments = []
            for i, comment_str in enumerate(finnish_comments):
                print(f"Translating Finnish comments {i+1}/{len(finnish_comments)}")
                if comment_str and comment_str != "":
                    translated_comment = self.translate_text(comment_str, 'fi')
                    translated_comments.append(translated_comment)
                else:
                    translated_comments.append("")
            
            df.loc[finnish_mask, 'translated_internal_comments'] = translated_comments
        
        # Translate Swedish texts
        swedish_mask = df['detected_language'] == 'sv'
        if swedish_mask.any():
            print(f"Translating {swedish_mask.sum()} Swedish tickets...")
            
            # Translate subjects
            swedish_subjects = df.loc[swedish_mask, 'subject'].tolist()
            translated_subjects = []
            for i, subject in enumerate(swedish_subjects):
                print(f"Translating Swedish subject {i+1}/{len(swedish_subjects)}: {subject[:50]}...")
                translated = self.translate_text(subject, 'sv')
                translated_subjects.append(translated)
            df.loc[swedish_mask, 'translated_subject'] = translated_subjects
            
            # Translate bodies
            swedish_bodies = df.loc[swedish_mask, 'body'].tolist()
            translated_bodies = []
            for i, body in enumerate(swedish_bodies):
                print(f"Translating Swedish body {i+1}/{len(swedish_bodies)}: {body[:50]}...")
                translated = self.translate_text(body, 'sv')
                translated_bodies.append(translated)
            df.loc[swedish_mask, 'translated_body'] = translated_bodies
            
            # Translate internal comments
            swedish_comments = df.loc[swedish_mask, 'internal_comments'].tolist()
            translated_comments = []
            for i, comment_str in enumerate(swedish_comments):
                print(f"Translating Swedish comments {i+1}/{len(swedish_comments)}")
                if comment_str and comment_str != "":
                    translated_comment = self.translate_text(comment_str, 'sv')
                    translated_comments.append(translated_comment)
                else:
                    translated_comments.append("")
            
            df.loc[swedish_mask, 'translated_internal_comments'] = translated_comments
        
        # Create structured text columns for ML
        df['customer_query_text'] = (
            df['translated_subject'].fillna('').astype(str) + " " + 
            df['translated_body'].fillna('').astype(str)
        )
        df['agent_actions_text'] = df['translated_internal_comments'].fillna('').astype(str)
        df['full_conversation_text'] = (
            "CUSTOMER: " + df['customer_query_text'] + " AGENT: " + df['agent_actions_text']
        )
        
        # Create original text column for reference
        df['original_combined_text'] = (
            df['subject'].fillna('').astype(str) + " " + 
            df['body'].fillna('').astype(str) + " " + 
            df['internal_comments'].fillna('').astype(str)
        )
        
        return df

    def process_dataframe(self, df):
        """Main method to process entire dataframe"""
        print(f"Dataset shape: {df.shape}")
        
        # Create copies of original data for traceability
        df['original_subject'] = df['subject']
        df['original_body'] = df['body']
        df['original_internal_comments'] = df['internal_comments']
        
        # Mask PII in original columns
        print("PII masking in subject and body...")
        df['subject'] = df['subject'].apply(self.mask_pii)
        df['body'] = df['body'].apply(self.mask_pii)

        # Apply to any text column
        df['subject'] = df['subject'].apply(self.remove_emojis)
        df['body'] = df['body'].apply(self.remove_emojis)
        
        print("PII masking in internal comments...")
        # Convert internal_comments to strings first
        df['internal_comments'] = df['internal_comments'].apply(self.process_internal_comments)
        
        # Translate non-English text
        df = self.batch_translate_dataframe(df)
        
        # Clean up only temporary column, keep detected_language
        df = df.drop(['combined_text'], axis=1, errors='ignore')
        
        print("Data processing completed!")
        return df

def main():
    data_path = "/home/faiz/Documents/github/Customer_Support/data/enfuce_support_tickets_synthetic_sample.jsonl"
    data = pd.read_json(data_path, lines=True)

    print(data.head())
    print(f"\nDataset shape: {data.shape}")

    # Initialize cleaner and process data
    cleaner = TextCleaner()
    cleaned_data = cleaner.process_dataframe(data)

    print("Cleaning completed!")
    
    # Show language distribution from the kept column
    print(f"Language distribution:\n{cleaned_data['detected_language'].value_counts()}")
    
    # Show sample of processed data
    print("\nSample processed tickets:")
    for i in range(min(5, len(cleaned_data))):
        print("*" * 80)
        row = cleaned_data.iloc[i]
        print(f"Ticket: {row['ticket_id']}")
        print(f"Original Language: {row['detected_language']}")
        print(f"Original Subject: {row['original_subject']}")
        print(f"Translated Subject: {row['translated_subject']}")
        print(f"Translation Quality: {'GOOD' if row['translated_subject'] != row['original_subject'] else 'SAME/POOR'}")
        print("*" * 80)

    # Save cleaned data
    cleaned_data.to_csv('/home/faiz/Documents/github/Customer_Support/data/customer_support_tickets.csv', index=False)
    print("Cleaned data saved to CSV!")

if __name__ == "__main__":
    main()