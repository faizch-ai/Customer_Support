# utils/text_cleaner.py

import pandas as pd
import re
from langdetect import detect
from .translate import LanguageTranslator

class TextCleaner:
    def __init__(self):
        self.translator = LanguageTranslator()
        self.PII_PATTERNS = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\+?[\d\s\-()]{10,}',
            'iban': r'[A-Z]{2}\d{2}[\sA-Z0-9]{10,30}',
            'card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }

    def detect_language(self, text):
        """Detect language using langdetect with proper error handling"""
        if pd.isna(text) or text == '':
            return 'unknown'
        try:
            lang = detect(str(text))
            if lang in ["no", "da", "de"]:
                lang = "sv"
            return lang
        except:
            return 'unknown'

    def mask_pii(self, text):
        """Mask PII in text"""
        if pd.isna(text):
            return text
            
        text = re.sub(self.PII_PATTERNS['email'], '[EMAIL]', text)
        text = re.sub(self.PII_PATTERNS['phone'], '[PHONE]', text)
        text = re.sub(self.PII_PATTERNS['iban'], '[IBAN]', text)
        text = re.sub(self.PII_PATTERNS['card'], '[CARD]', text)
        
        return text

    def clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return text
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def batch_translate_dataframe(self, df):
        """Batch translate all non-English content efficiently"""
        # Initialize translated columns
        df['translated_subject'] = df['clean_subject']
        df['translated_body'] = df['clean_body']
        df['translated_internal_comments'] = df['clean_internal_comments']
        
        # Process Finnish
        finnish_mask = df['detected_language'] == 'fi'
        if finnish_mask.any():
            print(f"Translating {finnish_mask.sum()} Finnish tickets...")
            
            finnish_df = df[finnish_mask]
            
            # Batch translate subjects
            finnish_subjects = list(finnish_df['clean_subject'])
            df.loc[finnish_mask, 'translated_subject'] = self.translator.fi_to_en(finnish_subjects)
            
            # Batch translate bodies
            finnish_bodies = list(finnish_df['clean_body'])
            df.loc[finnish_mask, 'translated_body'] = self.translator.fi_to_en(finnish_bodies)
            
            # Batch translate internal comments
            finnish_comments = list(finnish_df['clean_internal_comments'])
            translated_comments = []
            for comment_list in finnish_comments:
                if comment_list:
                    translated_comment_list = self.translator.fi_to_en(comment_list)
                    translated_comments.append(translated_comment_list)
                else:
                    translated_comments.append([])
            df.loc[finnish_mask, 'translated_internal_comments'] = translated_comments
        
        # Process Swedish
        swedish_mask = df['detected_language'] == 'sv'
        if swedish_mask.any():
            print(f"Translating {swedish_mask.sum()} Swedish tickets...")
            
            swedish_df = df[swedish_mask]
            
            # Batch translate subjects
            swedish_subjects = list(swedish_df['clean_subject'])
            df.loc[swedish_mask, 'translated_subject'] = self.translator.sv_to_en(swedish_subjects)
            
            # Batch translate bodies
            swedish_bodies = list(swedish_df['clean_body'])
            df.loc[swedish_mask, 'translated_body'] = self.translator.sv_to_en(swedish_bodies)
            
            # Batch translate internal comments
            swedish_comments = list(swedish_df['clean_internal_comments'])
            translated_comments = []
            for comment_list in swedish_comments:
                if comment_list:
                    translated_comment_list = self.translator.sv_to_en(comment_list)
                    translated_comments.append(translated_comment_list)
                else:
                    translated_comments.append([])
            df.loc[swedish_mask, 'translated_internal_comments'] = translated_comments
        
        return df

    def process_dataframe(self, df):
        """Main cleaning pipeline for the dataframe"""
        # Clean and mask PII
        for field in ['subject', 'body']:
            df[f'clean_{field}'] = df[field].apply(
                lambda x: self.mask_pii(self.clean_text(x))
            )
        
        print("PII masked in subject and body...")
        
        # Handle internal_comments (list of strings)
        df['clean_internal_comments'] = df['internal_comments'].apply(
            lambda x: [self.mask_pii(self.clean_text(comment)) for comment in x] 
            if isinstance(x, list) else []
        )

        print("PII masked in internal comments...")

        # Detect language
        df['detected_language'] = df['clean_body'].apply(self.detect_language)
        print("Language detection completed...")
        print(f"Language distribution:\n{df['detected_language'].value_counts()}")

        # Batch translate all non-English content
        df = self.batch_translate_dataframe(df)
        print("Batch translation completed...")

        # Create ML-ready combined text
        df['ml_ready_text'] = (
            df['translated_subject'] + " " + 
            df['translated_body'] + " " + 
            df['translated_internal_comments'].apply(
                lambda x: ' '.join(x) if x else ''
            )
        )

        print("ML-ready text created...")
        return df