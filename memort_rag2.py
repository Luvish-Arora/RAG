import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
import re
import PyPDF2
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from collections import defaultdict
from datetime import datetime

MODEL_ID = "mistralai/Mistral-7B-v0.1"
PEFT_ID = "Moodoxyy/mistral-7b-dsa-finetuned"

# ============================================================================
# TOPIC CLASSIFIER - Auto-detects topics from content
# ============================================================================

class TopicClassifier:
    """Automatically classifies content into topics using keyword matching."""
    
    def __init__(self):
        # Dynamic keyword database - can be extended for any domain
        self.keyword_database = {
            'database': ['database', 'sql', 'query', 'table', 'dbms', 'nosql', 'mysql', 
                        'postgresql', 'mongodb', 'redis', 'cassandra', 'schema', 'index',
                        'normalization', 'acid', 'transaction', 'join', 'foreign key'],
            'data_structures': ['array', 'linked list', 'tree', 'graph', 'stack', 'queue',
                               'heap', 'hash', 'binary tree', 'bst', 'avl', 'trie', 'list'],
            'algorithms': ['sort', 'search', 'algorithm', 'complexity', 'big o', 'time complexity',
                          'space complexity', 'dynamic programming', 'greedy', 'divide and conquer',
                          'recursion', 'iteration', 'optimization', 'bubble sort', 'merge sort'],
            'networking': ['network', 'tcp', 'ip', 'http', 'protocol', 'osi', 'dns', 'routing',
                          'socket', 'packet', 'ethernet', 'wifi', 'bandwidth', 'latency'],
            'operating_systems': ['process', 'thread', 'memory', 'operating system', 'os',
                                 'scheduling', 'deadlock', 'semaphore', 'mutex', 'virtual memory',
                                 'paging', 'kernel', 'system call', 'cpu', 'cache'],
            'programming': ['function', 'variable', 'loop', 'class', 'object', 'oop',
                           'inheritance', 'polymorphism', 'encapsulation', 'method',
                           'constructor', 'destructor', 'interface', 'abstract'],
            'web_development': ['html', 'css', 'javascript', 'react', 'angular', 'vue',
                               'node', 'express', 'rest', 'api', 'frontend', 'backend'],
            'machine_learning': ['neural network', 'deep learning', 'supervised', 'unsupervised',
                                'regression', 'classification', 'training', 'model', 'gradient'],
            'security': ['encryption', 'cryptography', 'authentication', 'authorization',
                        'firewall', 'vulnerability', 'attack', 'malware', 'https', 'ssl']
        }
    
    def classify(self, text):
        """
        Classify text into one or more topics.
        Returns primary topic and confidence score.
        """
        if not text:
            return 'general', 0.0
        
        text_lower = text.lower()
        
        # Count matches for each topic
        topic_scores = {}
        for topic, keywords in self.keyword_database.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                # Normalize by number of keywords in topic
                topic_scores[topic] = matches / len(keywords)
        
        if not topic_scores:
            return 'general', 0.0
        
        # Get topic with highest score
        primary_topic = max(topic_scores, key=topic_scores.get)
        confidence = topic_scores[primary_topic]
        
        return primary_topic, confidence
    
    def classify_detailed(self, text):
        """
        Get all topics present in text with scores.
        Returns list of (topic, score) tuples.
        """
        if not text:
            return [('general', 0.0)]
        
        text_lower = text.lower()
        topic_scores = []
        
        for topic, keywords in self.keyword_database.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                score = matches / len(keywords)
                topic_scores.append((topic, score))
        
        # Sort by score descending
        topic_scores.sort(key=lambda x: x[1], reverse=True)
        
        return topic_scores if topic_scores else [('general', 0.0)]
    
    def add_topic(self, topic_name, keywords):
        """Dynamically add new topic to the classifier."""
        self.keyword_database[topic_name] = keywords
        print(f"✅ Added new topic: {topic_name} with {len(keywords)} keywords")
    
    def get_all_topics(self):
        """Get list of all available topics."""
        return list(self.keyword_database.keys())


# ============================================================================
# ADAPTIVE LEARNING ENGINE
# ============================================================================

class AdaptiveLearningEngine:
    """Tracks user performance across ANY content/PDF."""
    
    def __init__(self):
        self.topic_classifier = TopicClassifier()
        self.user_responses = []
        self.topic_performance = defaultdict(lambda: {
            'correct': 0, 
            'total': 0, 
            'questions': [],
            'chunks': []  # Store related chunks
        })
        self.weak_topics = []
        self.strong_topics = []
        self.session_history = []
    
    def record_response(self, question, user_answer, correct_answer, 
                       question_text=None, chunk_source=None):
        """
        Record user response and auto-classify topic.
        No manual topic needed - auto-detected!
        """
        is_correct = (user_answer.upper() == correct_answer.upper())
        
        # Auto-classify topic from question
        full_text = f"{question} {question_text or ''}"
        primary_topic, confidence = self.topic_classifier.classify(full_text)
        all_topics = self.topic_classifier.classify_detailed(full_text)
        
        response = {
            'question': question,
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'primary_topic': primary_topic,
            'all_topics': all_topics,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        self.user_responses.append(response)
        
        # Update performance for all detected topics
        for topic, score in all_topics[:2]:  # Top 2 topics
            self.topic_performance[topic]['total'] += 1
            if is_correct:
                self.topic_performance[topic]['correct'] += 1
            self.topic_performance[topic]['questions'].append(question)
            if chunk_source:
                self.topic_performance[topic]['chunks'].append(chunk_source)
        
        print(f"📝 Question classified as: {primary_topic} (confidence: {confidence:.2f})")
        
        return response
    
    def analyze_performance(self):
        """Analyze performance and identify weak/strong areas."""
        weak_topics = []
        strong_topics = []
        medium_topics = []
        
        for topic, stats in self.topic_performance.items():
            if stats['total'] == 0:
                continue
            
            accuracy = stats['correct'] / stats['total']
            
            performance = {
                'topic': topic,
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total'],
                'sample_questions': stats['questions'][:3],
                'related_chunks': stats['chunks'][:3]
            }
            
            # Classify by accuracy thresholds
            if accuracy < 0.6:
                weak_topics.append(performance)
            elif accuracy > 0.8:
                strong_topics.append(performance)
            else:
                medium_topics.append(performance)
        
        # Sort by accuracy
        weak_topics.sort(key=lambda x: x['accuracy'])
        strong_topics.sort(key=lambda x: x['accuracy'], reverse=True)
        
        self.weak_topics = weak_topics
        self.strong_topics = strong_topics
        
        return {
            'weak_topics': weak_topics,
            'medium_topics': medium_topics,
            'strong_topics': strong_topics,
            'overall_accuracy': self._calculate_overall_accuracy()
        }
    
    def _calculate_overall_accuracy(self):
        """Calculate overall accuracy."""
        if not self.user_responses:
            return 0.0
        correct = sum(1 for r in self.user_responses if r['is_correct'])
        return correct / len(self.user_responses)
    
    def get_weak_topic_keywords(self):
        """Get keywords related to weak topics for RAG retrieval."""
        weak_keywords = []
        for topic_info in self.weak_topics:
            topic = topic_info['topic']
            if topic in self.topic_classifier.keyword_database:
                weak_keywords.extend(self.topic_classifier.keyword_database[topic])
        return weak_keywords
    
    def get_focus_distribution(self):
        """Get distribution for adaptive question generation."""
        return {
            'weak_weight': 0.7,
            'medium_weight': 0.2,
            'strong_weight': 0.1
        }
    
    def print_analysis(self):
        """Print detailed performance analysis."""
        analysis = self.analyze_performance()
        
        print("\n" + "="*70)
        print("📊 ADAPTIVE LEARNING ANALYSIS")
        print("="*70)
        print(f"Overall Accuracy: {analysis['overall_accuracy']*100:.1f}%")
        print(f"Total Questions Answered: {len(self.user_responses)}")
        
        print(f"\n❌ WEAK AREAS ({len(analysis['weak_topics'])} topics) - Need Focus:")
        for topic in analysis['weak_topics']:
            print(f"  • {topic['topic'].replace('_', ' ').title()}: "
                  f"{topic['accuracy']*100:.1f}% ({topic['correct']}/{topic['total']})")
        
        print(f"\n⚠️  MEDIUM AREAS ({len(analysis['medium_topics'])} topics) - Good Progress:")
        for topic in analysis['medium_topics']:
            print(f"  • {topic['topic'].replace('_', ' ').title()}: "
                  f"{topic['accuracy']*100:.1f}% ({topic['correct']}/{topic['total']})")
        
        print(f"\n✅ STRONG AREAS ({len(analysis['strong_topics'])} topics) - Mastered:")
        for topic in analysis['strong_topics']:
            print(f"  • {topic['topic'].replace('_', ' ').title()}: "
                  f"{topic['accuracy']*100:.1f}% ({topic['correct']}/{topic['total']})")
        
        print("="*70 + "\n")
    
    def save_session(self, filename="learning_session.json"):
        """Save learning session data."""
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'responses': self.user_responses,
            'weak_topics': self.weak_topics,
            'strong_topics': self.strong_topics,
            'overall_accuracy': self._calculate_overall_accuracy(),
            'topic_performance': dict(self.topic_performance)
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"💾 Session saved to {filename}")
    
    def load_session(self, filename="learning_session.json"):
        """Load previous learning session to continue progress."""
        try:
            with open(filename, 'r') as f:
                session_data = json.load(f)
            
            self.user_responses = session_data.get('responses', [])
            self.topic_performance = defaultdict(lambda: {
                'correct': 0, 'total': 0, 'questions': [], 'chunks': []
            }, session_data.get('topic_performance', {}))
            
            print(f"✅ Loaded previous session from {filename}")
            print(f"   Continuing from {len(self.user_responses)} previous responses")
            return True
        except FileNotFoundError:
            print(f"ℹ️  No previous session found. Starting fresh.")
            return False


# ============================================================================
# ADAPTIVE RAG RETRIEVER
# ============================================================================

class AdaptiveRAGRetriever:
    """RAG retriever that adapts based on user's weak topics."""
    
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        print(f"\n🔍 Initializing Adaptive RAG Retriever...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.chunk_topics = []
        self.topic_classifier = TopicClassifier()
        print("✅ Adaptive RAG Retriever initialized")
    
    def build_index(self, chunks):
        """Build FAISS index and classify all chunks."""
        print(f"\n📚 Building vector index from {len(chunks)} chunks...")
        self.chunks = chunks
        
        # Classify each chunk
        print("🏷️  Classifying chunks by topic...")
        self.chunk_topics = []
        for i, chunk in enumerate(chunks):
            primary_topic, confidence = self.topic_classifier.classify(chunk)
            self.chunk_topics.append({
                'primary_topic': primary_topic,
                'confidence': confidence
            })
            if (i + 1) % 10 == 0:
                print(f"   Classified {i + 1}/{len(chunks)} chunks...")
        
        # Generate embeddings
        print("🔄 Generating embeddings...")
        self.embeddings = self.embedding_model.encode(
            chunks, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        print(f"✅ Index built with {len(chunks)} chunks")
        
        # Print topic distribution
        topic_counts = defaultdict(int)
        for chunk_topic in self.chunk_topics:
            topic_counts[chunk_topic['primary_topic']] += 1
        
        print(f"\n📊 Topic Distribution in PDF:")
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {topic.replace('_', ' ').title()}: {count} chunks")
    
    def retrieve_adaptive(self, query, learning_engine, top_k=5):
        """
        Retrieve chunks with adaptive weighting.
        Boosts chunks from weak topics automatically!
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Get weak topics from learning engine
        weak_topic_names = [t['topic'] for t in learning_engine.weak_topics]
        
        if weak_topic_names:
            print(f"🎯 Boosting retrieval for weak areas: {weak_topic_names}")
        
        # Retrieve more candidates
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, min(top_k * 3, len(self.chunks)))
        
        # Score chunks: base relevance + weak topic bonus
        scored_chunks = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            chunk_topic = self.chunk_topics[idx]['primary_topic']
            
            # Base score (lower distance = better)
            score = -float(distance)
            
            # BOOST: Add bonus if chunk is from weak area
            if chunk_topic in weak_topic_names:
                score += 15.0  # Significant boost for weak topics!
                print(f"   ⬆️  Boosted chunk from weak topic: {chunk_topic}")
            
            scored_chunks.append((chunk, score, chunk_topic, idx))
        
        # Sort by score and select top-k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        selected_chunks = [(chunk, idx) for chunk, score, topic, idx in scored_chunks[:top_k]]
        
        return selected_chunks
    
    def retrieve_for_weak_areas(self, learning_engine, top_k=5):
        """Retrieve chunks specifically targeting weak areas."""
        
        # Get weak topic keywords
        weak_keywords = learning_engine.get_weak_topic_keywords()
        
        if not weak_keywords:
            # No weak areas yet, return diverse chunks
            print("ℹ️  No weak areas identified yet. Selecting diverse chunks...")
            indices = np.random.choice(len(self.chunks), min(top_k, len(self.chunks)), replace=False)
            return [(self.chunks[i], i) for i in indices]
        
        # Create query from weak keywords
        query = " ".join(weak_keywords[:10])  # Use top 10 keywords
        print(f"🔍 Searching for chunks related to: {', '.join(weak_keywords[:5])}...")
        
        return self.retrieve_adaptive(query, learning_engine, top_k)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load fine-tuned Mistral model."""
    print("\n" + "="*70)
    print("🧠 Loading Moodoxyy/mistral-7b-dsa-finetuned")
    print("="*70)
    
    print("\n📦 Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("✅ Tokenizer loaded")

    print("\n📦 Step 2: Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    print("✅ Base model loaded")
    print("\n📦 Step 3: Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(model, PEFT_ID)
    model.eval()
    
    print(f"✅ Model ready!")
    print("="*70 + "\n")
    
    return model, tokenizer


# ============================================================================
# PDF PROCESSING
# ============================================================================

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    print(f"\n📄 Extracting text from: {pdf_path}")
    
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            print(f"📖 Total pages: {num_pages}")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            
            print(f"✅ Extracted {len(text)} characters")
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return None
    
    return text


def chunk_text(text, chunk_size=2000, overlap=200):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size // 2:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


# ============================================================================
# MCQ GENERATION - Standard (Round 1)
# ============================================================================

def create_standard_prompt(main_content, related_contexts, difficulty="medium", num_questions=1):
    """Create standard prompt WITHOUT topic restrictions."""
    
    context_summary = "\n\n".join([f"Context {i+1}:\n{ctx[:400]}..." 
                                   for i, ctx in enumerate(related_contexts)])
    
    prompt = f"""Generate {num_questions} high-quality multiple choice question(s) from this content.

Main Content:
{main_content}

Related Information:
{context_summary}

Difficulty: {difficulty}

Create questions that test understanding, not just memorization.

Format:
Question: [question text]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
Correct Answer: [A/B/C/D]
Explanation: [explanation]

---
"""
    
    return prompt


def generate_standard_mcqs_unlimited(model, tokenizer, retriever, difficulty="medium", 
                                     questions_per_chunk=1):
    """
    Generate MCQs from ALL chunks (no limit).
    
    Args:
        questions_per_chunk: Number of questions to generate per chunk (default: 1)
    """
    
    print("\n" + "="*70)
    print("📝 Generating Initial Assessment Questions from ALL Chunks")
    print("="*70)
    print(f"📊 Processing {len(retriever.chunks)} chunks...")
    
    all_questions = []
    
    for i, chunk_idx in enumerate(range(len(retriever.chunks))):
        chunk = retriever.chunks[chunk_idx]
        print(f"\n📝 Processing chunk {i+1}/{len(retriever.chunks)}...")
        
        # Get related contexts
        query_embedding = retriever.embedding_model.encode([chunk], convert_to_numpy=True)
        distances, indices = retriever.index.search(query_embedding, 4)
        related = [retriever.chunks[idx] for idx in indices[0][1:4]]  # Skip self
        
        # Create prompt
        prompt = create_standard_prompt(
            main_content=chunk[:1200],
            related_contexts=related,
            difficulty=difficulty,
            num_questions=questions_per_chunk
        )
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        mcq_text = generated[len(prompt):].strip()
        
        # Parse
        questions = parse_mcqs(mcq_text)
        questions = clean_and_validate_mcqs(questions)
        
        # Add source chunk info
        for q in questions:
            q['source_chunk_idx'] = chunk_idx
        
        all_questions.extend(questions)
        
        if questions:
            print(f"✅ Generated {len(questions)} valid question(s) from this chunk")
            print(f"📊 Total questions so far: {len(all_questions)}")
        else:
            print(f"⚠️  No valid questions from this chunk")
    
    print(f"\n{'='*70}")
    print(f"🎉 Generation complete! Total valid questions: {len(all_questions)}")
    print(f"{'='*70}")
    
    return all_questions

# ============================================================================
# MCQ GENERATION - Adaptive (Round 2+)
# ============================================================================

def create_adaptive_prompt(main_content, related_contexts, learning_engine, 
                          difficulty="medium", num_questions=1):
    """Create adaptive prompt focusing on weak areas."""
    
    weak_topics = [t['topic'].replace('_', ' ').title() for t in learning_engine.weak_topics]
    
    focus_instruction = ""
    if weak_topics:
        focus_instruction = f"\n🎯 PRIORITY: Create challenging questions about: {', '.join(weak_topics)}"
        focus_instruction += "\nThese are areas where the learner needs more practice."
    
    context_summary = "\n\n".join([f"Context {i+1}:\n{ctx[:400]}..." 
                                   for i, ctx in enumerate(related_contexts)])
    
    prompt = f"""Generate {num_questions} targeted multiple choice question(s) for adaptive learning.

Main Content:
{main_content}

Related Information:
{context_summary}

{focus_instruction}

Difficulty: {difficulty}

Create questions that help improve understanding in weak areas.

Format:
Question: [question text]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
Correct Answer: [A/B/C/D]
Explanation: [detailed explanation]

---
"""
    
    return prompt


def generate_adaptive_mcqs_unlimited(model, tokenizer, retriever, learning_engine, 
                                    difficulty="medium", weak_chunks_percentage=0.7):
    """
    Generate adaptive MCQs from ALL weak-area chunks (no limit).
    
    Args:
        weak_chunks_percentage: Percentage of chunks to focus on weak areas (0.7 = 70%)
    """
    
    print("\n" + "="*70)
    print("🎯 Generating Adaptive Questions from ALL Weak-Area Chunks")
    print("="*70)
    
    weak_topic_names = [t['topic'] for t in learning_engine.weak_topics]
    
    if not weak_topic_names:
        print("ℹ️  No weak areas yet, using all chunks...")
        target_chunks = list(range(len(retriever.chunks)))
    else:
        print(f"🎯 Targeting weak topics: {weak_topic_names}")
        
        # Find all chunks related to weak topics
        weak_chunks = []
        medium_chunks = []
        
        for idx, chunk_topic_info in enumerate(retriever.chunk_topics):
            if chunk_topic_info['primary_topic'] in weak_topic_names:
                weak_chunks.append(idx)
            else:
                medium_chunks.append(idx)
        
        print(f"📊 Found {len(weak_chunks)} chunks in weak areas")
        print(f"📊 Found {len(medium_chunks)} chunks in other areas")
        
        # Select chunks based on percentage
        num_weak = int(len(weak_chunks) * 1.0)  # Use all weak chunks
        num_medium = int(len(medium_chunks) * (1 - weak_chunks_percentage))
        
        target_chunks = weak_chunks[:num_weak] + medium_chunks[:num_medium]
        
        print(f"📊 Will generate questions from {len(target_chunks)} chunks")
        print(f"   - {num_weak} from weak areas (priority)")
        print(f"   - {num_medium} from other areas (maintenance)")
    
    all_questions = []
    
    for i, chunk_idx in enumerate(target_chunks):
        chunk = retriever.chunks[chunk_idx]
        chunk_topic = retriever.chunk_topics[chunk_idx]['primary_topic']
        
        print(f"\n📝 Processing chunk {i+1}/{len(target_chunks)} (Topic: {chunk_topic})...")
        
        # Get related contexts with adaptive boost
        related_chunks = retriever.retrieve_adaptive(chunk, learning_engine, top_k=3)
        related = [c for c, idx in related_chunks]
        
        # Create adaptive prompt
        prompt = create_adaptive_prompt(
            main_content=chunk[:1200],
            related_contexts=related,
            learning_engine=learning_engine,
            difficulty=difficulty,
            num_questions=1
        )
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        mcq_text = generated[len(prompt):].strip()
        
        # Parse
        questions = parse_mcqs(mcq_text)
        questions = clean_and_validate_mcqs(questions)
        
        # Add source info
        for q in questions:
            q['source_chunk_idx'] = chunk_idx
            q['chunk_topic'] = chunk_topic
        
        all_questions.extend(questions)
        
        if questions:
            print(f"✅ Generated {len(questions)} valid question(s)")
            print(f"📊 Total questions so far: {len(all_questions)}")
        else:
            print(f"⚠️  No valid questions from this chunk")
    
    print(f"\n{'='*70}")
    print(f"🎉 Adaptive generation complete! Total questions: {len(all_questions)}")
    print(f"{'='*70}")
    
    return all_questions


# ============================================================================
# PARSING & VALIDATION
# ============================================================================

def parse_mcqs(mcq_text):
    """Parse MCQ text into structured format."""
    questions = []
    question_blocks = mcq_text.split("---")
    
    for block in question_blocks:
        block = block.strip()
        if not block:
            continue
        
        mcq = {}
        
        q_match = re.search(r'Question:\s*(.+?)(?=\n[A-D]\))', block, re.DOTALL)
        if q_match:
            mcq['question'] = q_match.group(1).strip()
        
        options = {}
        for letter in ['A', 'B', 'C', 'D']:
            opt_pattern = rf'{letter}\)\s*(.+?)(?=\n[A-D]\)|\nCorrect Answer:|\nAnswer:|\nExplanation:|$)'
            opt_match = re.search(opt_pattern, block, re.DOTALL)
            
            if opt_match:
                option_text = opt_match.group(1).strip()
                option_text = re.sub(r'\n?Answer:\s*[A-D]\).*$', '', option_text, flags=re.DOTALL).strip()
                options[letter] = option_text
        
        mcq['options'] = options
        
        ans_match = re.search(r'Correct Answer:\s*([A-D])', block, re.IGNORECASE)
        if ans_match:
            mcq['correct_answer'] = ans_match.group(1).strip().upper()
        else:
            ans_match = re.search(r'Answer:\s*([A-D])\)?', block, re.IGNORECASE)
            if ans_match:
                mcq['correct_answer'] = ans_match.group(1).strip().upper()
        
        exp_match = re.search(r'Explanation:\s*(.+?)(?=\n\nQuestion:|\n---|\Z)', block, re.DOTALL | re.IGNORECASE)
        if exp_match:
            mcq['explanation'] = exp_match.group(1).strip()
        
        if mcq.get('question') and mcq.get('options'):
            if 'correct_answer' not in mcq:
                mcq['correct_answer'] = list(mcq['options'].keys())[0]
            if 'explanation' not in mcq or not mcq['explanation']:
                mcq['explanation'] = "No explanation provided."
            
            questions.append(mcq)
    
    return questions


def clean_and_validate_mcqs(questions):
    """Clean and validate MCQs."""
    cleaned = []
    
    for i, mcq in enumerate(questions, 1):
        if 'question' not in mcq or len(mcq['question']) < 20:
            continue
        
        if len(mcq.get('options', {})) < 2:
            continue
        
        cleaned_mcq = {
            'question': mcq['question'].strip(),
            'options': {k: v.strip() for k, v in mcq['options'].items()},
            'correct_answer': mcq.get('correct_answer', 'A'),
            'explanation': mcq.get('explanation', 'No explanation provided.'),
            'source_chunk_idx': mcq.get('source_chunk_idx')
        }
        
        cleaned.append(cleaned_mcq)
    
    return cleaned


# ============================================================================
# INTERACTIVE QUIZ
# ============================================================================
def conduct_batch_quiz(questions, learning_engine, retriever, batch_size=10):
    """
    Conduct quiz in batches with progress saving.
    
    Args:
        batch_size: Number of questions to answer before saving progress
    """
    
    print("\n" + "="*70)
    print(f"📝 BATCH QUIZ MODE - {len(questions)} Total Questions")
    print("="*70)
    print(f"Questions will be presented in batches of {batch_size}")
    print("Progress is saved after each batch\n")
    
    total_batches = (len(questions) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(questions))
        batch_questions = questions[start_idx:end_idx]
        
        print(f"\n{'='*70}")
        print(f"📦 BATCH {batch_num + 1}/{total_batches}")
        print(f"Questions {start_idx + 1} to {end_idx} of {len(questions)}")
        print(f"{'='*70}\n")
        
        for i, mcq in enumerate(batch_questions):
            question_num = start_idx + i + 1
            
            print(f"\nQuestion {question_num}/{len(questions)}:")
            print(mcq['question'])
            print()
            
            for letter, option in mcq['options'].items():
                print(f"{letter}) {option}")
            
            # Get user answer
            while True:
                user_answer = input("\nYour answer (A/B/C/D or 'skip' to skip): ").strip().upper()
                if user_answer in ['A', 'B', 'C', 'D', 'SKIP']:
                    break
                print("❌ Invalid input. Please enter A, B, C, D, or 'skip'.")
            
            if user_answer == 'SKIP':
                print("⏭️  Skipped")
                continue
            
            # Get source chunk
            chunk_source = None
            if mcq.get('source_chunk_idx') is not None:
                chunk_source = retriever.chunks[mcq['source_chunk_idx']]
            
            # Record response
            learning_engine.record_response(
                question=mcq['question'],
                user_answer=user_answer,
                correct_answer=mcq['correct_answer'],
                question_text=" ".join(mcq['options'].values()),
                chunk_source=chunk_source
            )
            
            # Show feedback
            if user_answer == mcq['correct_answer']:
                print("✅ Correct!")
            else:
                print(f"❌ Incorrect. The correct answer is {mcq['correct_answer']}")
            
            print(f"💡 Explanation: {mcq['explanation']}")
            print("-" * 70)
        
        # Save progress after each batch
        learning_engine.save_session(f"session_batch_{batch_num + 1}.json")
        print(f"\n💾 Progress saved! Completed {end_idx}/{len(questions)} questions")
        
        # Show intermediate analysis
        if batch_num < total_batches - 1:
            print("\n📊 Current Performance:")
            analysis = learning_engine.analyze_performance()
            print(f"Overall Accuracy: {analysis['overall_accuracy']*100:.1f}%")
            
            continue_quiz = input("\nContinue to next batch? (yes/no): ").strip().lower()
            if continue_quiz != 'yes' and continue_quiz != 'y':
                print("\n⏸️  Quiz paused. Progress saved. Run again to continue.")
                return False
    
    print("\n✅ All questions completed!")
    return True

def conduct_interactive_quiz(questions, learning_engine, retriever):
    """Conduct interactive quiz with auto topic detection."""
    
    print("\n" + "="*70)
    print("📝 INTERACTIVE QUIZ")
    print("="*70)
    print("Answer each question by typing A, B, C, or D\n")
    
    for i, mcq in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}:")
        print(mcq['question'])
        print()
        
        for letter, option in mcq['options'].items():
            print(f"{letter}) {option}")
        
        # Get user answer
        while True:
            user_answer = input("\nYour answer (A/B/C/D): ").strip().upper()
            if user_answer in ['A', 'B', 'C', 'D']:
                break
            print("❌ Invalid input. Please enter A, B, C, or D.")
        
        # Get source chunk for context
        chunk_source = None
        if mcq.get('source_chunk_idx') is not None:
            chunk_source = retriever.chunks[mcq['source_chunk_idx']]
        
        # Record response (auto-detects topic!)
        learning_engine.record_response(
            question=mcq['question'],
            user_answer=user_answer,
            correct_answer=mcq['correct_answer'],
            question_text=" ".join(mcq['options'].values()),
            chunk_source=chunk_source
        )
        
        # Show feedback
        if user_answer == mcq['correct_answer']:
            print("✅ Correct!")
        else:
            print(f"❌ Incorrect. The correct answer is {mcq['correct_answer']}")
        
        print(f"💡 Explanation: {mcq['explanation']}")
        print("-" * 70)
    
    print("\n✅ Quiz completed!")


# ============================================================================
# MAIN ADAPTIVE PIPELINE
# ============================================================================
def adaptive_learning_pipeline_unlimited(pdf_path, num_rounds=2, 
                                        continue_session=False, batch_mode=True):
    """
    Adaptive learning pipeline with NO question limits.
    Generates questions from ALL chunks in the PDF.
    
    Args:
        pdf_path: Path to PDF file
        num_rounds: Number of learning rounds
        continue_session: Load previous session
        batch_mode: Use batch quiz mode (recommended for many questions)
    """
    
    print("\n" + "="*70)
    print("🚀 UNLIMITED ADAPTIVE LEARNING SYSTEM")
    print("="*70)
    print("📚 Generates questions from ALL chunks - No limits!")
    print("="*70)
    
    # Load model
    model, tokenizer = load_model()
    
    # Extract and index PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        return
    
    chunks = chunk_text(pdf_text, chunk_size=2000, overlap=200)
    print(f"\n📊 Total chunks to process: {len(chunks)}")
    
    # Initialize components
    retriever = AdaptiveRAGRetriever()
    retriever.build_index(chunks)
    
    learning_engine = AdaptiveLearningEngine()
    
    # Optionally continue previous session
    if continue_session:
        learning_engine.load_session()
    
    # Multiple rounds
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*70}")
        print(f"📚 ROUND {round_num}/{num_rounds}")
        print(f"{'='*70}")
        
        if round_num == 1 and len(learning_engine.user_responses) == 0:
            # First round: all chunks
            print("\n🎯 Generating questions from ALL chunks...")
            print("   (This may take a while for large PDFs)")
            
            questions = generate_standard_mcqs_unlimited(
                model, tokenizer, retriever,
                difficulty="medium",
                questions_per_chunk=1  # 1 question per chunk
            )
        else:
            # Subsequent rounds: focus on weak areas
            learning_engine.print_analysis()
            
            print("\n🎯 Generating questions focusing on weak areas...")
            questions = generate_adaptive_mcqs_unlimited(
                model, tokenizer, retriever, learning_engine,
                difficulty="medium",
                weak_chunks_percentage=0.7  # 70% from weak areas
            )
        
        if not questions:
            print("\n⚠️  No questions generated. Ending session.")
            break
        
        # Conduct quiz
        if batch_mode:
            completed = conduct_batch_quiz(questions, learning_engine, retriever, batch_size=10)
            if not completed:
                print("\n💾 Session saved. Run with continue_session=True to resume.")
                break
        else:
            conduct_interactive_quiz(questions, learning_engine, retriever)
    
    # Final analysis
    print("\n" + "="*70)
    print("🎓 FINAL PERFORMANCE REPORT")
    print("="*70)
    learning_engine.print_analysis()
    
    # Save final session
    learning_engine.save_session("final_session.json")
    
    return learning_engine


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Option 1: UNLIMITED MODE - Generate from ALL chunks
    learning_engine = adaptive_learning_pipeline_unlimited(
        pdf_path="sample1.pdf",
        num_rounds=2,
        continue_session=False,
        batch_mode=True  # Recommended for many questions
    )
    
    # Option 2: Original LIMITED mode - For quick testing
    # learning_engine = adaptive_learning_pipeline(
    #     pdf_path="your_document.pdf",
    #     num_rounds=2,
    #     questions_per_round=5,  # Limited to 5 per round
    #     continue_session=False
    # )
    
    print("\n🎉 Learning session complete!")