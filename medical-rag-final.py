"""
✅ FIXED MEDICAL RAG SYSTEM - RESOLVES EMBEDDING MODEL ERROR
Apollo 2B + Advanced Medical RAG (All fixes applied)

Date: 2025-11-16
Status: Fixed - Uses compatible embedding model

ISSUE FIXED:
- allenai/specter2 was causing PEFT config error
- Switched to all-MiniLM-L6-v2 (compatible) + better alternatives
- Code now handles multiple embedding models gracefully
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import requests
import re

# ============================================================================
# CONFIGURATION - ALL OPTIMIZATIONS APPLIED
# ============================================================================

class AdvancedConfig:
    """✅ OPTIMIZED Configuration for Medical Domain"""
    
    # Paths
    ADAPTER_PATH = "./apollo2b-medical-qa-final"
    RAG_LIBRARY_DIR = "./medical_rag_library"
    RAG_LIBRARY_NAME = "medical_rag"
    FEEDBACK_FILE = "./medical_rag_library/user_feedback.json"
    
    # ✅ FIX 1: EMBEDDING MODEL (FIXED - Use compatible model)
    # If you get embedding errors, these are the best options:
    # Option 1 (Fast, compatible): all-MiniLM-L6-v2
    # Option 2 (Better, requires more memory): sentence-transformers/all-mpnet-base-v2
    # Option 3 (Medical, if available): allenai/specter2_base
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ✅ FIXED - Use this if specter2 fails
    # EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Alternative (better)
    
    # ✅ FIX 2: MEDICAL RERANKER
    RERANKER_MODEL = "cross-encoder/mmarco-MiniLM-L12-v2"
    
    # ✅ FIX 3: RETRIEVAL - OPTIMIZED
    TOP_K_RETRIEVAL = 15  # Increased from 10
    TOP_K_RERANK = 8      # Increased from 5
    MIN_SIMILARITY = 0.35 # Increased from 0.20 (CRITICAL!)
    
    # ✅ FIX 4: GENERATION - MORE CONSERVATIVE
    MAX_NEW_TOKENS = 150     # Increased from 120
    TEMPERATURE = 0.5        # Decreased from 0.6
    TOP_P = 0.85            # Decreased from 0.9
    REPETITION_PENALTY = 1.5 # Increased from 1.3
    
    # ✅ FIX 5: CHUNKING PARAMETERS
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 100
    
    # Features
    ENABLE_WEB_SEARCH = True
    USE_COT = True
    USE_RERANKING = True
    CONTEXT_WINDOW = 5


# ============================================================================
# HELPER: SMART EMBEDDING MODEL LOADER (WITH FALLBACK)
# ============================================================================

def load_embedding_model_safe(model_name: str):
    """
    ✅ FIXED: Load embedding model with fallback options
    Handles various embedding model formats
    """
    print(f"📦 Attempting to load embedding model: {model_name}...")
    
    try:
        # Try to load the specified model
        model = SentenceTransformer(model_name)
        print(f"✅ Successfully loaded: {model_name}")
        return model
    except Exception as e:
        print(f"⚠️ Warning: Could not load {model_name}")
        print(f"   Error: {str(e)[:100]}")
        
        # Fallback option 1: Use simpler model name
        if "specter2" in model_name.lower():
            print("   Trying fallback: all-MiniLM-L6-v2")
            try:
                model = SentenceTransformer("all-MiniLM-L6-v2")
                print("✅ Loaded fallback: all-MiniLM-L6-v2")
                return model
            except Exception as e2:
                print(f"⚠️ Fallback failed: {e2}")
        
        # Fallback option 2: Use mpnet
        print("   Trying fallback: sentence-transformers/all-mpnet-base-v2")
        try:
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            print("✅ Loaded fallback: all-mpnet-base-v2")
            return model
        except Exception as e3:
            print(f"⚠️ Fallback failed: {e3}")
        
        # Final fallback: Use tiny model
        print("   Trying final fallback: all-MiniLM-L6-v2")
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Loaded final fallback: all-MiniLM-L6-v2")
            return model
        except Exception as e4:
            print(f"❌ All fallbacks failed: {e4}")
            raise


# ============================================================================
# FEEDBACK STORE
# ============================================================================

class FeedbackStore:
    """Store and retrieve user feedback for learning"""
    
    def __init__(self, feedback_file: str):
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_data = self.load_feedback()
    
    def load_feedback(self) -> Dict:
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except:
                return {"corrections": [], "preferences": {}}
        return {"corrections": [], "preferences": {}}
    
    def save_feedback(self):
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def add_correction(self, question: str, wrong_answer: str, correct_answer: str):
        correction = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "wrong_answer": wrong_answer,
            "correct_answer": correct_answer
        }
        self.feedback_data["corrections"].append(correction)
        self.save_feedback()
        print(f"✅ Feedback saved! Total: {len(self.feedback_data['corrections'])}")
    
    def get_relevant_corrections(self, query: str, max_results: int = 3) -> List[Dict]:
        corrections = []
        query_lower = query.lower()
        for correction in self.feedback_data["corrections"]:
            if any(word in correction["question"].lower() for word in query_lower.split()):
                corrections.append(correction)
        return corrections[-max_results:]


# ============================================================================
# WEB SEARCH HELPER
# ============================================================================

class WebSearchHelper:
    """Free web search for medical information"""
    
    @staticmethod
    def search_pubmed(query: str, max_results: int = 3) -> List[str]:
        try:
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json"
            search_resp = requests.get(search_url, timeout=5)
            search_data = search_resp.json()
            pmids = search_data.get('esearchresult', {}).get('idlist', [])
            if not pmids:
                return []
            fetch_url = f"{base_url}esummary.fcgi?db=pubmed&id={','.join(pmids)}&retmode=json"
            fetch_resp = requests.get(fetch_url, timeout=5)
            fetch_data = fetch_resp.json()
            results = []
            for pmid in pmids:
                if pmid in fetch_data.get('result', {}):
                    article = fetch_data['result'][pmid]
                    title = article.get('title', '')
                    source = article.get('source', '')
                    results.append(f"{title} ({source})")
            return results
        except:
            return []
    
    @staticmethod
    def search_duckduckgo(query: str) -> List[str]:
        try:
            url = f"https://api.duckduckgo.com/?q={query} medical&format=json&no_html=1"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            results = []
            if data.get('AbstractText'):
                results.append(data['AbstractText'][:300])
            for topic in data.get('RelatedTopics', [])[:2]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append(topic['Text'][:200])
            return results
        except:
            return []


# ============================================================================
# LOCAL RAG LOADER (FIXED)
# ============================================================================

class LocalRAGLoader:
    def __init__(self, library_dir: str, library_name: str, embedding_model_name: str):
        self.library_dir = Path(library_dir)
        
        print(f"📚 Loading RAG library...")
        
        metadata_file = self.library_dir / f"{library_name}_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"❌ Metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # ✅ FIXED: Use safe embedding loader with fallback
        self.embedding_model = load_embedding_model_safe(embedding_model_name)
        
        index_file = self.library_dir / f"{library_name}_index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"❌ Index not found: {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        print(f"✅ FAISS index loaded ({self.index.ntotal} vectors)")
        
        docs_file = self.library_dir / f"{library_name}_documents.pkl"
        if not docs_file.exists():
            raise FileNotFoundError(f"❌ Documents not found: {docs_file}")
        
        with open(docs_file, 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"✅ Loaded {len(self.documents)} documents")


# ============================================================================
# ADVANCED VECTOR STORE (OPTIMIZED)
# ============================================================================

class AdvancedVectorStore:
    """✅ OPTIMIZED Vector Store with better retrieval"""
    
    def __init__(self, local_rag: LocalRAGLoader, reranker_model: Optional[str] = None, min_sim: float = 0.35):
        self.embedding_model = local_rag.embedding_model
        self.index = local_rag.index
        self.documents = local_rag.documents
        self.min_sim = min_sim
        
        self.reranker = None
        if reranker_model:
            print(f"🔨 Loading reranker: {reranker_model}...")
            try:
                self.reranker = CrossEncoder(reranker_model)
                print("✅ Reranker loaded")
            except Exception as e:
                print(f"⚠️ Warning: Could not load reranker: {e}")
                print("   Continuing without reranking")
    
    def retrieve_and_rerank(self, query: str, top_k: int = 15, rerank_top_k: int = 8, use_reranking: bool = True) -> List[Tuple[Dict, float]]:
        """✅ OPTIMIZED: Better retrieval with improved thresholds"""
        
        if self.index.ntotal == 0:
            return []
        
        try:
            query_emb = self.embedding_model.encode([query], normalize_embeddings=True).astype('float32')
            sims, indices = self.index.search(query_emb, min(top_k, self.index.ntotal))
            
            candidates = [
                (self.documents[idx], float(sim))
                for idx, sim in zip(indices[0], sims[0])
                if idx != -1 and idx < len(self.documents) and float(sim) >= self.min_sim
            ]
            
            if not candidates:
                return []
            
            if self.reranker and use_reranking:
                try:
                    texts = [doc["text"] for doc, _ in candidates]
                    pairs = [(query, text) for text in texts]
                    rerank_scores = self.reranker.predict(pairs)
                    
                    combined = []
                    for i, (doc, orig_score) in enumerate(candidates):
                        combined_score = 0.4 * orig_score + 0.6 * rerank_scores[i]
                        combined.append((doc, combined_score))
                    
                    combined.sort(key=lambda x: x[1], reverse=True)
                    return combined[:rerank_top_k]
                except Exception as e:
                    print(f"⚠️ Reranking failed: {e}, using original scores")
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:rerank_top_k]
        
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []


# ============================================================================
# MAIN MEDICAL RAG SYSTEM (OPTIMIZED)
# ============================================================================

class AdvancedMedicalRAG:
    """✅ OPTIMIZED Medical RAG with all improvements"""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.web_search = WebSearchHelper()
        self.feedback_store = FeedbackStore(config.FEEDBACK_FILE)
        self.conversation_history = []
        
        print("\n" + "="*70)
        print("🚀 OPTIMIZED MEDICAL RAG - CONVERSATIONAL WITH LEARNING")
        print("="*70)
        print("✅ All fixes applied:")
        print(f"   • Embedding: {config.EMBEDDING_MODEL}")
        print(f"   • Min Similarity: {config.MIN_SIMILARITY}")
        print(f"   • Top-K: {config.TOP_K_RETRIEVAL}, Rerank: {config.TOP_K_RERANK}")
        print(f"   • Temperature: {config.TEMPERATURE}")
        print(f"   • Chunk Size: {config.CHUNK_SIZE}")
        print("="*70)
        
        print(f"\n📦 Loading model...")
        try:
            peft_config = PeftConfig.from_pretrained(config.ADAPTER_PATH)
            base_model_path = peft_config.base_model_name_or_path
            
            print(f"   Base model: {base_model_path}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.model = PeftModel.from_pretrained(self.model, config.ADAPTER_PATH)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"✓ Model loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        print(f"\n📚 Loading RAG library...")
        try:
            local_rag = LocalRAGLoader(config.RAG_LIBRARY_DIR, config.RAG_LIBRARY_NAME, config.EMBEDDING_MODEL)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            raise
        
        print(f"\n🔨 Initializing vector store...")
        self.vector_store = AdvancedVectorStore(local_rag, config.RERANKER_MODEL, config.MIN_SIMILARITY)
        
        print(f"\n📝 Loaded {len(self.feedback_store.feedback_data['corrections'])} corrections")
        
        print("\n✅ SYSTEM READY!")
        print("="*70)
    
    def clear_context(self):
        self.conversation_history = []
    
    def get_context_string(self) -> str:
        if not self.conversation_history:
            return ""
        recent = self.conversation_history[-self.config.CONTEXT_WINDOW:]
        return "\n".join([f"User: {msg['question']}\nAssistant: {msg['answer']}" for msg in recent])
    
    def create_prompt(self, question: str, context: str, corrections: List[Dict], conversation_context: str, use_cot: bool = True) -> str:
        corrections_str = ""
        if corrections:
            corrections_str = "\n\nIMPORTANT - Past User Corrections:\n"
            for corr in corrections:
                corrections_str += f"Q: {corr['question']}\nCorrect: {corr['correct_answer']}\n\n"
        
        conv_str = ""
        if conversation_context:
            conv_str = f"\n\nPrevious: {conversation_context}\n"
        
        if use_cot:
            return f"""You are a medical expert. Answer based on evidence and corrections.

{corrections_str}

Evidence:
{context}
{conv_str}

Question: {question}

Answer (2-3 sentences):"""
        else:
            return f"""Answer based on evidence.

{corrections_str}

Evidence:
{context}
{conv_str}

Question: {question}

Answer:"""
    
    def generate_with_confidence(self, question: str) -> Dict:
        """✅ OPTIMIZED: Generate with better parameters"""
        
        corrections = self.feedback_store.get_relevant_corrections(question)
        
        results = self.vector_store.retrieve_and_rerank(
            question,
            top_k=self.config.TOP_K_RETRIEVAL,
            rerank_top_k=self.config.TOP_K_RERANK,
            use_reranking=self.config.USE_RERANKING
        )
        
        web_results = []
        
        if not results or (len(results) > 0 and np.mean([score for _, score in results]) < 0.35):
            if self.config.ENABLE_WEB_SEARCH:
                print("🌐 Searching web...")
                web_results.extend(self.web_search.search_pubmed(question, max_results=3))
                web_results.extend(self.web_search.search_duckduckgo(question))
        
        context_parts = []
        source_list = []
        
        for i, web_result in enumerate(web_results[:3], 1):
            context_parts.append(f"[WEB {i}] {web_result}")
            source_list.append({'text': web_result, 'score': 0.8, 'type': 'web'})
        
        for i, (doc, score) in enumerate(results[:2], len(context_parts) + 1):
            context_parts.append(f"[Source {i}] {doc['text'][:300]}")
            source_list.append({'text': doc['text'], 'score': score, 'type': 'local'})
        
        if not context_parts and not corrections:
            return {
                "question": question,
                "answer": "Insufficient information. Consult healthcare professional.",
                "confidence": 0.0,
                "sources": [],
                "method": "no_info",
                "web_results": 0
            }
        
        context = "\n\n".join(context_parts)
        conversation_context = self.get_context_string()
        prompt = self.create_prompt(question, context, corrections, conversation_context, self.config.USE_COT)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=self.config.REPETITION_PENALTY,
                no_repeat_ngram_size=3
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Answer:" in full_text:
            answer = full_text.split("Answer:")[-1].strip()
        else:
            answer = full_text[len(prompt):].strip()
        
        answer = re.sub(r'\[.*?\]', '', answer)
        answer = re.sub(r'(Explanation|Key|Common):', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        sentences = [s.strip() + '.' for s in answer.split('.') if s.strip() and len(s.strip()) > 20]
        final_answer = ' '.join(sentences[:4]) if sentences else answer
        
        if results:
            confidence = float(np.mean([score for _, score in results]))
        else:
            confidence = 0.3
        
        if len(web_results) > 0:
            confidence = min(confidence * 1.3, 1.0)
        
        self.conversation_history.append({
            "question": question,
            "answer": final_answer,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "question": question,
            "answer": final_answer,
            "confidence": confidence,
            "sources": source_list,
            "method": "hybrid" if len(web_results) > 0 else "local_rag",
            "web_results": len(web_results),
            "corrections_used": len(corrections)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "*"*70)
    print("INITIALIZING OPTIMIZED MEDICAL RAG SYSTEM")
    print("*"*70)
    
    try:
        # Initialize
        config = AdvancedConfig()
        rag_system = AdvancedMedicalRAG(config)
        
        # Quick test
        print("\n" + "="*70)
        print("QUICK TEST")
        print("="*70)
        
        test_questions = [
            "What are the symptoms of diabetes?",
            "How is hypertension treated?",
        ]
        
        for question in test_questions:
            print(f"\n🔹 Q: {question}")
            try:
                result = rag_system.generate_with_confidence(question)
                print(f"✓ A: {result['answer']}")
                print(f"   Confidence: {result['confidence']:.1%}")
                print(f"   Method: {result['method']}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n✅ Done! System ready for use.")
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if adapter exists: ./apollo2b-medical-qa-final")
        print("2. Check if RAG library exists: ./medical_rag_library/")
        print("3. Verify RAG files:")
        print("   - medical_rag_metadata.json")
        print("   - medical_rag_index.faiss")
        print("   - medical_rag_documents.pkl")
        import traceback
        traceback.print_exc()