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
# ADAPTIVE LEARNING ENGINE
# ============================================================================

class AdaptiveLearningEngine:
    """Tracks user performance and identifies weak/strong areas."""
    
    def __init__(self):
        self.user_responses = []
        self.topic_performance = defaultdict(lambda: {'correct': 0, 'total': 0, 'questions': []})
        self.weak_topics = []
        self.strong_topics = []
        self.session_history = []
    
    def record_response(self, question, user_answer, correct_answer, topic=None):
        """Record a user's response to a question."""
        is_correct = (user_answer.upper() == correct_answer.upper())
        
        response = {
            'question': question,
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'topic': topic or self._extract_topic(question),
            'timestamp': datetime.now().isoformat()
        }
        
        self.user_responses.append(response)
        
        # Update topic performance
        topic_key = response['topic']
        self.topic_performance[topic_key]['total'] += 1
        if is_correct:
            self.topic_performance[topic_key]['correct'] += 1
        self.topic_performance[topic_key]['questions'].append(question)
        
        return response
    
    def _extract_topic(self, question):
        """Extract topic from question using keywords."""
        # Simple keyword-based topic extraction
        keywords = {
            'database': ['database', 'sql', 'query', 'table', 'dbms'],
            'data_structures': ['array', 'linked list', 'tree', 'graph', 'stack', 'queue'],
            'algorithms': ['sort', 'search', 'algorithm', 'complexity', 'time'],
            'networking': ['network', 'tcp', 'ip', 'http', 'protocol'],
            'os': ['process', 'thread', 'memory', 'operating system'],
            'programming': ['function', 'variable', 'loop', 'class', 'object']
        }
        
        question_lower = question.lower()
        for topic, words in keywords.items():
            if any(word in question_lower for word in words):
                return topic
        
        return 'general'
    
    def analyze_performance(self):
        """Analyze user performance and identify weak/strong areas."""
        weak_topics = []
        strong_topics = []
        
        for topic, stats in self.topic_performance.items():
            if stats['total'] == 0:
                continue
            
            accuracy = stats['correct'] / stats['total']
            
            performance = {
                'topic': topic,
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total'],
                'sample_questions': stats['questions'][:3]
            }
            
            # Classify as weak (< 60%) or strong (> 80%)
            if accuracy < 0.6:
                weak_topics.append(performance)
            elif accuracy > 0.8:
                strong_topics.append(performance)
        
        # Sort by accuracy
        weak_topics.sort(key=lambda x: x['accuracy'])
        strong_topics.sort(key=lambda x: x['accuracy'], reverse=True)
        
        self.weak_topics = weak_topics
        self.strong_topics = strong_topics
        
        return {
            'weak_topics': weak_topics,
            'strong_topics': strong_topics,
            'overall_accuracy': self._calculate_overall_accuracy()
        }
    
    def _calculate_overall_accuracy(self):
        """Calculate overall accuracy across all responses."""
        if not self.user_responses:
            return 0.0
        
        correct = sum(1 for r in self.user_responses if r['is_correct'])
        return correct / len(self.user_responses)
    
    def get_weak_contexts(self):
        """Get text contexts related to weak topics."""
        weak_contexts = []
        for topic_info in self.weak_topics:
            weak_contexts.extend(topic_info['sample_questions'])
        return weak_contexts
    
    def get_focus_distribution(self):
        """Get distribution for adaptive question generation."""
        total_weak = len(self.weak_topics)
        total_strong = len(self.strong_topics)
        
        # Allocate 70% to weak, 20% to medium, 10% to strong
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
        
        print(f"\n❌ Weak Areas ({len(analysis['weak_topics'])} topics):")
        for topic in analysis['weak_topics']:
            print(f"  • {topic['topic']}: {topic['accuracy']*100:.1f}% "
                  f"({topic['correct']}/{topic['total']})")
        
        print(f"\n✅ Strong Areas ({len(analysis['strong_topics'])} topics):")
        for topic in analysis['strong_topics']:
            print(f"  • {topic['topic']}: {topic['accuracy']*100:.1f}% "
                  f"({topic['correct']}/{topic['total']})")
        
        print("="*70 + "\n")
    
    def save_session(self, filename="learning_session.json"):
        """Save learning session data."""
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'responses': self.user_responses,
            'weak_topics': self.weak_topics,
            'strong_topics': self.strong_topics,
            'overall_accuracy': self._calculate_overall_accuracy()
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"💾 Session saved to {filename}")


# ============================================================================
# ADAPTIVE RAG RETRIEVER
# ============================================================================

class AdaptiveRAGRetriever:
    """RAG retriever with adaptive weighting based on user performance."""
    
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        print(f"\n🔍 Initializing Adaptive RAG Retriever...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.chunk_topics = []  # Track topic for each chunk
        print("✅ Adaptive RAG Retriever initialized")
    
    def build_index(self, chunks):
        """Build FAISS index from text chunks."""
        print(f"\n📚 Building vector index from {len(chunks)} chunks...")
        self.chunks = chunks
        
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
        
        # Assign topics to chunks (simple keyword matching)
        self.chunk_topics = [self._classify_chunk(chunk) for chunk in chunks]
        
        print(f"✅ Index built with {len(chunks)} chunks")
    
    def _classify_chunk(self, chunk):
        """Classify chunk into topic categories."""
        keywords = {
            'database': ['database', 'sql', 'query', 'table', 'dbms', 'nosql'],
            'data_structures': ['array', 'linked list', 'tree', 'graph', 'stack', 'queue'],
            'algorithms': ['sort', 'search', 'algorithm', 'complexity', 'big o'],
            'networking': ['network', 'tcp', 'ip', 'http', 'protocol'],
            'os': ['process', 'thread', 'memory', 'operating system'],
            'programming': ['function', 'variable', 'loop', 'class', 'object']
        }
        
        chunk_lower = chunk.lower()
        for topic, words in keywords.items():
            if any(word in chunk_lower for word in words):
                return topic
        
        return 'general'
    
    def retrieve_adaptive(self, query, learning_engine, top_k=5):
        """Retrieve chunks with adaptive weighting based on weak areas."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Get weak topics
        weak_topics = [t['topic'] for t in learning_engine.weak_topics]
        
        print(f"🎯 Focusing on weak areas: {weak_topics}")
        
        # Retrieve more candidates
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k * 3)
        
        # Score chunks based on relevance + weak topic bonus
        scored_chunks = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            chunk_topic = self.chunk_topics[idx]
            
            # Base score (lower distance = better)
            score = -distance
            
            # Boost score if chunk is from weak area
            if chunk_topic in weak_topics:
                score += 10.0  # Significant boost for weak topics
            
            scored_chunks.append((chunk, score, chunk_topic))
        
        # Sort by score and select top-k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        selected_chunks = [chunk for chunk, score, topic in scored_chunks[:top_k]]
        selected_topics = [topic for chunk, score, topic in scored_chunks[:top_k]]
        
        print(f"✅ Retrieved chunks focusing on: {set(selected_topics)}")
        
        return selected_chunks
    
    def retrieve_for_weak_areas(self, learning_engine, top_k=5):
        """Retrieve chunks specifically for weak areas."""
        weak_contexts = learning_engine.get_weak_contexts()
        
        if not weak_contexts:
            # No weak areas yet, return random chunks
            return np.random.choice(self.chunks, min(top_k, len(self.chunks)), replace=False).tolist()
        
        # Combine weak context queries
        combined_query = " ".join(weak_contexts[:3])
        
        return self.retrieve_adaptive(combined_query, learning_engine, top_k)


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
    
    print(f"✅ Model ready on device: {model.device}")
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
# ADAPTIVE MCQ GENERATION
# ============================================================================

def create_adaptive_prompt(main_content, related_contexts, learning_engine, 
                          difficulty="medium", num_questions=1):
    """Create prompt adapted to user's weak areas."""
    
    weak_topics = [t['topic'] for t in learning_engine.weak_topics] if learning_engine.weak_topics else []
    strong_topics = [t['topic'] for t in learning_engine.strong_topics] if learning_engine.strong_topics else []
    
    # Build focus instruction
    focus_instruction = ""
    if weak_topics:
        focus_instruction = f"\nIMPORTANT: Focus primarily on these weak areas: {', '.join(weak_topics)}. "
        focus_instruction += "Generate challenging questions that test deep understanding of these topics."
    
    context_summary = "\n\n".join([f"Context {i+1}:\n{ctx[:400]}..." 
                                   for i, ctx in enumerate(related_contexts)])
    
    prompt = f"""You are creating personalized assessment questions based on learner performance.

Main Content:
{main_content}

Related Information:
{context_summary}

Learner Profile:
- Weak areas: {', '.join(weak_topics) if weak_topics else 'Not yet determined'}
- Strong areas: {', '.join(strong_topics) if strong_topics else 'Not yet determined'}

{focus_instruction}

Task: Generate {num_questions} targeted multiple choice question(s).
Difficulty: {difficulty}

Requirements:
- Focus on concepts the learner struggles with
- Create challenging but fair questions
- Test application and analysis, not just recall
- Ensure all options are plausible

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


def generate_adaptive_mcqs(model, tokenizer, retriever, learning_engine, 
                          num_questions=5, difficulty="medium"):
    """Generate MCQs adapted to user's performance."""
    
    print("\n" + "="*70)
    print("🎯 Generating Adaptive Questions")
    print("="*70)
    
    # Retrieve chunks focused on weak areas
    print("\n🔍 Retrieving content for weak areas...")
    weak_chunks = retriever.retrieve_for_weak_areas(learning_engine, top_k=5)
    
    all_questions = []
    
    for i, chunk in enumerate(weak_chunks[:num_questions]):
        print(f"\n📝 Generating question {i+1}/{num_questions}...")
        
        # Get related contexts
        related = retriever.retrieve_adaptive(chunk, learning_engine, top_k=3)
        
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
        
        all_questions.extend(questions)
        print(f"✅ Generated {len(questions)} valid question(s)")
    
    return all_questions


# ============================================================================
# PARSING & VALIDATION (Same as before)
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
            'explanation': mcq.get('explanation', 'No explanation provided.')
        }
        
        cleaned.append(cleaned_mcq)
    
    return cleaned


# ============================================================================
# INTERACTIVE QUIZ
# ============================================================================

def conduct_interactive_quiz(questions, learning_engine):
    """Conduct an interactive quiz and record responses."""
    
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
        
        # Record response
        learning_engine.record_response(
            question=mcq['question'],
            user_answer=user_answer,
            correct_answer=mcq['correct_answer']
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
# MAIN PIPELINE
# ============================================================================

def adaptive_learning_pipeline(pdf_path, num_rounds=2, questions_per_round=5):
    """Complete adaptive learning pipeline with multiple rounds."""
    
    print("\n" + "="*70)
    print("🚀 ADAPTIVE LEARNING SYSTEM")
    print("="*70)
    
    # Load model
    model, tokenizer = load_model()
    
    # Extract and index PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        return
    
    chunks = chunk_text(pdf_text, chunk_size=2000, overlap=200)
    
    # Initialize components
    retriever = AdaptiveRAGRetriever()
    retriever.build_index(chunks)
    
    learning_engine = AdaptiveLearningEngine()
    
    # Multiple rounds
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*70}")
        print(f"📚 ROUND {round_num}/{num_rounds}")
        print(f"{'='*70}")
        
        if round_num == 1:
            # First round: general questions
            print("\n🎯 Generating initial assessment questions...")
            questions = generate_adaptive_mcqs(
                model, tokenizer, retriever, learning_engine,
                num_questions=questions_per_round,
                difficulty="medium"
            )
        else:
            # Analyze and adapt
            learning_engine.print_analysis()
            
            print("\n🎯 Generating adaptive questions based on your performance...")
            questions = generate_adaptive_mcqs(
                model, tokenizer, retriever, learning_engine,
                num_questions=questions_per_round,
                difficulty="medium"
            )
        
        # Conduct quiz
        conduct_interactive_quiz(questions, learning_engine)
    
    # Final analysis
    print("\n" + "="*70)
    print("🎓 FINAL PERFORMANCE REPORT")
    print("="*70)
    learning_engine.print_analysis()
    
    # Save session
    learning_engine.save_session()
    
    return learning_engine


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Run adaptive learning system
    pdf_path = "sample1.pdf"  # Replace with your PDF
    
    learning_engine = adaptive_learning_pipeline(
        pdf_path=pdf_path,
        num_rounds=2,  # Start with 2 rounds
        questions_per_round=5  # 5 questions per round
    )
    
    print("\n🎉 Adaptive learning session complete!")