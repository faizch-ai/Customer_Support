# support_agent_demo.py

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch
import re
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import warnings
warnings.filterwarnings('ignore')

# === FAISS additions ===
import os
from pathlib import Path
import faiss

class SupportAgentAssistant:
    def __init__(self):
        st.set_page_config(page_title="Enfuce Support AI", layout="wide")
        self._inject_theme_css()
        self._llm_ready = False
        self.textgen = None

        self.suggestion_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"        
        # Load models
        with st.spinner("Loading AI models..."):
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # --- LLM for suggestions: TinyLlama 1.1B Chat ---
        try:
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.suggestion_model_name = model_id

            tok = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True,
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cuda",           # or "auto"
                torch_dtype=torch.float16,   # safer than bf16 on many GPUs
                trust_remote_code=True,
            )

            # keep model/tokenizer aligned + padding defined
            if mdl.get_input_embeddings().weight.size(0) != len(tok):
                mdl.resize_token_embeddings(len(tok))
            if tok.pad_token_id is None:
                tok.pad_token_id = tok.eos_token_id
            mdl.config.pad_token_id = tok.pad_token_id

            def build_llama_inst(system_msg: str, user_msg: str) -> str:
                return (
                    "<s>[INST] <<SYS>>\n"
                    f"{system_msg.strip()}\n"
                    "<</SYS>>\n\n"
                    f"{user_msg.strip()}\n"
                    "[/INST]"
                )

            # one clean INST prompt; no apply_chat_template; no extra specials
            def tinyllama_chat(messages, max_new_tokens=80, temperature=0.0, top_p=1.0):
                sys = next((m["content"] for m in messages if m["role"] == "system"),
                        "You are a helpful assistant.")
                # latest user only (keeps output clean)
                latest_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
                prompt = build_llama_inst(sys, latest_user)

                enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
                enc = {k: v.to(mdl.device) for k, v in enc.items()}
                with torch.no_grad():
                    out = mdl.generate(
                        **enc,
                        max_new_tokens=120,           # give it a bit more room
                        do_sample=False,              # stable, deterministic
                        repetition_penalty=1.05,
                        no_repeat_ngram_size=3,
                        eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.pad_token_id,
                    )

                # Decode only the generated continuation (exclude the prompt)
                gen_ids = out[0][enc["input_ids"].shape[1]:]
                text = tok.decode(gen_ids, skip_special_tokens=True).strip()
                return text

            self.textgen = tinyllama_chat
            self._llm_ready = True

        except Exception as e:
            st.warning(
                f"Suggestion model unavailable ({self.suggestion_model_name}). "
                f"Skipping LLM suggestions.\n{e}"
            )
            self._llm_ready = False
            self.textgen = None
        
        # Known categories for classification
        self.categories = [
            'transaction issues', 'KYC verification', 'card management',
            'API integration issues', 'payment disputes', 'billing and invoices',
            'rate limits and quotas', 'technical errors'
        ]
        
        # Urgency keywords
        self.urgency_keywords = {
            'high': ['urgent', 'priority', 'prio', 'blocking', 'broken', 'down', 'critical', 'asap', 'emergency', 'hurry', 'quickly', 'fraudulent'],
            'medium': ['issue', 'problem', 'error', 'not working', 'failed', 'help needed'],
            'low': ['question', 'inquiry', 'info', 'clarification']
        }

        # === FAISS state ===
        self.faiss_cache_dir = Path(".faiss_cache")
        self.faiss_cache_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path = self.faiss_cache_dir / "kb.index"
        self.faiss_emb_path = self.faiss_cache_dir / "kb_embeddings.npy"
        self.faiss_meta_path = self.faiss_cache_dir / "kb_meta.npy"  # store row indices mapping

        self.index = None
        self.kb_embeddings = None
        self.kb_row_ids = None

    def _inject_theme_css(self):
        """Sidebar-only Enfuce styling; main area stays Streamlit default."""
        ENFUCE = {
            "navy":  "#0C0F25",
            "white": "#FFFFFF",
            "teal":  "#10A4BE",
            "aqua":  "#1ED4F2",
            "mint":  "#65E2B6",
            "lime":  "#C7F279",
        }
        st.markdown(f"""
        <style>
        /* === SIDEBAR ONLY === */
        [data-testid="stSidebar"] {{
            background:
            radial-gradient(80rem 50rem at -20% 110%, {ENFUCE["lime"]}55 0%, transparent 45%),
            radial-gradient(70rem 45rem at 10% 95%,  {ENFUCE["aqua"]}40 0%, transparent 55%),
            radial-gradient(75rem 50rem at 20% 85%,  {ENFUCE["teal"]}30 0%, transparent 60%),
            {ENFUCE["navy"]};
            border-right: 1px solid rgba(255,255,255,0.08);
        }}
        /* Sidebar text + headings */
        [data-testid="stSidebar"] * {{
            color: {ENFUCE["white"]} !important;
        }}
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {{
            color: {ENFUCE["white"]} !important;
            letter-spacing: 0.2px;
        }}
        /* Sidebar inputs/selects */
        [data-testid="stSidebar"] input, 
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [role="combobox"] {{
            background: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            color: {ENFUCE["white"]} !important;
            border-radius: 10px !important;
        }}
        /* Sidebar radio labels (keep them clearly white) */
        [data-testid="stSidebar"] [data-testid="stRadio"] label,
        [data-testid="stSidebar"] div[role="radiogroup"] * {{
            color: {ENFUCE["white"]} !important;
            opacity: 1 !important;
        }}
        /* Sidebar buttons */
        [data-testid="stSidebar"] .stButton > button {{
            background: linear-gradient(135deg, {ENFUCE["teal"]}, {ENFUCE["aqua"]});
            color: #0B1022;
            border: none;
            border-radius: 12px;
            padding: 0.5rem 0.9rem;
            font-weight: 600;
            box-shadow: 0 6px 18px rgba(30,212,242,0.25);
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{ filter: brightness(1.05); }}
        /* Sidebar links */
        [data-testid="stSidebar"] a {{ color: {ENFUCE["mint"]} !important; }}
        /* === MAIN AREA: no overrides (keeps default black/white) === */
        </style>
        """, unsafe_allow_html=True)

    
    def _build_and_cache_faiss(self, texts):
        """
        Encode texts, build FAISS index (cosine via normalized IP), and cache.
        """
        embeddings = self.model.encode(texts)
        embeddings = embeddings.astype('float32')

        # Normalize for cosine similarity with IndexFlatIP
        faiss.normalize_L2(embeddings)

        # Build index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save to disk
        faiss.write_index(index, str(self.faiss_index_path))
        np.save(self.faiss_emb_path, embeddings)
        # Row ids are simple direct mapping to dataframe indices 0..N-1
        row_ids = np.arange(len(texts), dtype=np.int64)
        np.save(self.faiss_meta_path, row_ids)

        self.index = index
        self.kb_embeddings = embeddings
        self.kb_row_ids = row_ids

    def _load_faiss_if_valid(self, expected_len):
        """
        Try loading FAISS index & cached embeddings if they match the dataset size.
        """
        try:
            if (
                self.faiss_index_path.exists() and
                self.faiss_emb_path.exists() and
                self.faiss_meta_path.exists()
            ):
                index = faiss.read_index(str(self.faiss_index_path))
                embeddings = np.load(self.faiss_emb_path)
                row_ids = np.load(self.faiss_meta_path)

                if embeddings.shape[0] == expected_len and row_ids.shape[0] == expected_len:
                    self.index = index
                    self.kb_embeddings = embeddings
                    self.kb_row_ids = row_ids
                    return True
        except Exception:
            pass
        return False

    def _pick_best_resolved(self, sims):
        """
        Choose the highest-scoring resolved ticket from the retrieved list.
        """
        best = None
        best_score = -1.0
        for t in sims:
            if str(t.get('status', '')).lower() == 'resolved':
                score = float(t.get('similarity_score', 0.0))
                if score > best_score:
                    best = t
                    best_score = score
        return best

    def _build_llm_prompt(self, selected_ticket, best_resolved, comments_text):
        subj = selected_ticket.get('translated_subject') or selected_ticket.get('subject') or ""
        body = selected_ticket.get('translated_body') or selected_ticket.get('body') or ""
        cat  = self.categorize_ticket(selected_ticket.get('customer_query_text', "")) or "Other"

        return (
            f"You are a senior support lead. Turn the following internal resolution notes into actionable, grammatically correct, sentence: {comments_text}"
        )

    def _generate_agent_suggestion(self, selected_ticket, best_resolved):
        if not self._llm_ready or not best_resolved:
            return None

        comments = (
            best_resolved.get('translated_internal_comments')
            or best_resolved.get('internal_comments')
            or ""
        ).strip()
        if not comments:
            return None

        system_msg = "You are a senior support lead. Be concise and actionable."
        user_msg = (
            "Rewrite these internal resolution notes as one clear, actionable sentence "
            "for the agent to send to the customer:\n"
            f"{comments}"
        )

        try:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            suggestion = self.textgen(messages).strip()
            return suggestion
        except Exception as e:
            st.warning(f"Suggestion generation failed: {e}")
            return None



    def load_data(self):
        """Load and prepare the dataset"""
        try:
            df = pd.read_csv('/home/faiz/Documents/github/Customer_Support/data/customer_support_tickets.csv')
            
            # Create demo split: 1% for query, 99% for knowledge base
            demo_size = max(1, len(df) // 100)  # 1% for demo
            self.demo_tickets = df.tail(demo_size).copy()  # Last 1% for selection
            self.knowledge_base = df.iloc[:-demo_size].copy()  # First 99% for similarity
            
            # Generate or load embeddings for knowledge base (FAISS)
            texts = self.knowledge_base['customer_query_text'].fillna('').tolist()

            # If cache valid, load; otherwise build and cache
            if not self._load_faiss_if_valid(len(texts)):
                with st.spinner("Building similarity index..."):
                    self._build_and_cache_faiss(texts)

            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def detect_urgency(self, text):
        """Detect urgency level from text"""
        text_lower = text.lower()
        
        high_count = sum(1 for word in self.urgency_keywords['high'] if word in text_lower)
        medium_count = sum(1 for word in self.urgency_keywords['medium'] if word in text_lower)
        low_count = sum(1 for word in self.urgency_keywords['low'] if word in text_lower)
        
        if high_count > 0:
            return "🚨 HIGH", "red"
        elif medium_count > 1:
            return "MEDIUM", "orange" 
        else:
            return "LOW", "green"
    
    def find_similar_tickets(self, query_text, top_k=10):
        """Find most similar tickets from knowledge base"""
        # Encode query
        query_embedding = self.model.encode([query_text]).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Scores are inner products (cosine because of normalization)
        D, I = self.index.search(query_embedding, top_k)
        scores = D[0]
        indices = I[0]

        similar_tickets = []
        for rank, idx in enumerate(indices):
            if idx < 0:
                continue  # FAISS returns -1 if not enough vectors
            ticket = self.knowledge_base.iloc[idx]
            similar_tickets.append({
                'similarity_score': float(scores[rank]),
                'ticket_id': ticket['ticket_id'],
                'subject': ticket['subject'],
                'translated_subject': ticket.get('translated_subject', ticket['subject']),
                'body_preview': ticket['body'],
                'translated_body_preview': ticket.get('translated_body', ticket['body']),
                'status': ticket['status'],
                'resolution_time': ticket['resolution_time_s'],
                'internal_comments': ticket['internal_comments'],
                'translated_internal_comments': ticket.get('translated_internal_comments', ticket['internal_comments']),
                'detected_language': ticket.get('detected_language', 'en'),
                'category': self.categorize_ticket(ticket['customer_query_text'])
            })
        
        return similar_tickets
    
    def categorize_ticket(self, text):
        """Simple rule-based categorization"""
        text_lower = text.lower()
        
        category_rules = {
            'transaction issues': ['transaction', 'payment', 'declined', 'failed payment'],
            'KYC verification': ['kyc', 'document', 'verify', 'identification'],
            'card management': ['card', 'lost', 'stolen', 'replace', 'pin'],
            'API integration issues': ['api', 'integration', 'endpoint', 'sdk'],
            'rate limits and quotas': ['rate limit', 'quota', 'throttle', '429'],
            'technical errors': ['error', 'bug', 'technical', 'crash'],
            'billing and invoices': ['invoice', 'billing', 'charge', 'fee'],
            'payment disputes': ['dispute', 'chargeback', 'fraud']
        }
        
        for category, keywords in category_rules.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return "Other"
    
    def run_demo(self):
        """Main demo interface"""
        st.title("Enfuce Support Agent Assistant")
        st.markdown("### AI-Powered Ticket Resolution Support")
        
        # Load data
        if not hasattr(self, 'demo_tickets'):
            if not self.load_data():
                return
        
        # Sidebar for ticket selection
        st.sidebar.header("Select a Ticket to Analyze")
        
        # Create ticket selection options with translated subject preview
        ticket_options = []
        for idx, ticket in self.demo_tickets.iterrows():
            # Use translated subject if available, otherwise original
            subject_preview = ticket.get('original_subject', ticket['subject'])
            preview = f"{ticket['ticket_id']}: {subject_preview[:50]}..."
            ticket_options.append((preview, idx))
        
        selected_option = st.sidebar.selectbox(
            "Choose a ticket:",
            options=[opt[0] for opt in ticket_options],
            index=0
        )
        
        selected_idx = [opt[1] for opt in ticket_options if opt[0] == selected_option][0]
        selected_ticket = self.demo_tickets.loc[selected_idx]
        
        # Display selected ticket with translations
        st.header("Selected Ticket")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show original and translated subject
            st.subheader("Original Subject")
            st.write(selected_ticket['subject'])
            
            if 'translated_subject' in selected_ticket and selected_ticket['translated_subject'] != selected_ticket['subject']:
                st.subheader("Translated Subject")
                st.info(selected_ticket['translated_subject'])
            
            # Show original and translated body
            st.subheader("Original Body")
            st.write(selected_ticket['body'])
            
            if 'translated_body' in selected_ticket and selected_ticket['translated_body'] != selected_ticket['body']:
                st.subheader("Translated Body")
                st.info(selected_ticket['translated_body'])
        
        with col2:
            # Urgency detection (using translated text for better accuracy)
            urgency_text = selected_ticket.get('translated_subject', selected_ticket['subject']) + " " + selected_ticket.get('translated_body', selected_ticket['body'])
            urgency_level, color = self.detect_urgency(urgency_text)
            st.markdown(f"### **Urgency: :{color}[{urgency_level}]**")
            
            # Category
            category = self.categorize_ticket(selected_ticket['customer_query_text'])
            st.markdown(f"**Category:** {category}")
            
            # Language info
            detected_lang = selected_ticket.get('detected_language', 'unknown')
            st.markdown(f"**Detected Language:** {detected_lang.upper()}")
            
            # Basic info
            st.markdown(f"**Ticket ID:** {selected_ticket['ticket_id']}")
            st.markdown(f"**Channel:** {selected_ticket['channel']}")
            st.markdown(f"**Status:** {selected_ticket['status']}")
        
        # Find similar tickets and generate recommendations
        with st.spinner("Finding similar cases and generating recommendations..."):
            query_text = selected_ticket['customer_query_text']  # This already uses translated content
            similar_tickets = self.find_similar_tickets(query_text)

        best_resolved = self._pick_best_resolved(similar_tickets)
        if best_resolved:
            suggestion = self._generate_agent_suggestion(selected_ticket, best_resolved)
            if suggestion:
                st.header("Suggested Next Step")
                st.success(suggestion)
                st.caption(f"Derived from resolved ticket: {best_resolved['ticket_id']} (score {best_resolved['similarity_score']:.3f})")
        
        # Display recommendations
        st.header("Recommended Actions")
        st.info("Based on analysis of similar historical tickets:")
        
        # Display similar tickets with translations
        st.header("Similar Historical Tickets")
        st.info(f"Found {len(similar_tickets)} similar tickets from knowledge base of {len(self.knowledge_base)} tickets")
        
        for i, similar in enumerate(similar_tickets[:5], 1):
            with st.expander(f"Similar Case #{i} (Score: {similar['similarity_score']:.3f}) - {similar['ticket_id']} - {similar['category']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Show subject comparison
                    st.write("**Original Subject:**")
                    st.write(similar['subject'])
                    
                    if similar['translated_subject'] != similar['subject']:
                        st.write("**Translated Subject:**")
                        st.info(similar['translated_subject'])
                    
                    # Show body preview comparison
                    st.write("**Body Preview:**")
                    st.write(similar['body_preview'])
                    
                    if similar['translated_body_preview'] != similar['body_preview']:
                        st.write("**Translated Preview:**")
                        st.info(similar['translated_body_preview'])
                    
                    # Show language info
                    if similar['detected_language'] != 'en':
                        st.write(f"**Originally in:** {similar['detected_language'].upper()}")
                
                with col2:
                    st.write(f"**Status:** {similar['status']}")
                    if pd.notna(similar['resolution_time']):
                        hours = similar['resolution_time'] / 3600
                        st.write(f"**Resolved in:** {hours:.1f}h")
                    
                    # Show if this was a successful resolution
                    if similar['status'] == 'resolved':
                        st.success("Successfully resolved")
                    
                    # Show agent actions from this similar case
                    if similar['translated_internal_comments']:
                        st.write("**Agent Actions:**")
                        st.text(similar['translated_internal_comments'])
        
        # Agent feedback section
        st.header("Agent Feedback")
        
        feedback = st.radio(
            "How helpful were these recommendations?",
            ["Very helpful", "Somewhat helpful", "Not helpful"]
        )

        urgence_feedback = st.radio(
            "Urgency false alarm? Please select the actual one.",
            ["HIGH", "MEDIUM", "LOW"]
        )

        category_feedback = st.text_area(
            "Was the predicted category correct? If not, please help us by adding the actual category in the field below",
            placeholder="transaction issues, KYC verification, card management, API integration issues, rate limits and quotas, technical errors, billing and invoices, payment disputes, or a new category?"
        )

        action_feedback = st.text_area(
            "If the suggested action didn't help, could you please enter what you actually did?"
        )
        
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback! This helps improve the system.")
        
        # Statistics
        st.sidebar.header("System Stats")
        st.sidebar.metric("Knowledge Base Tickets", len(self.knowledge_base))
        st.sidebar.metric("Demo Tickets", len(self.demo_tickets))
        
        # Language distribution in demo tickets
        if 'detected_language' in self.demo_tickets.columns:
            lang_dist = self.demo_tickets['detected_language'].value_counts()
            st.sidebar.header("🌍 Demo Tickets by Language")
            for lang, count in lang_dist.items():
                st.sidebar.text(f"{lang.upper()}: {count}")

def main():
    assistant = SupportAgentAssistant()
    assistant.run_demo()

if __name__ == "__main__":
    main()
