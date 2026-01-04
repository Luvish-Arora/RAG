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

MODEL_ID = "mistralai/Mistral-7B-v0.1"
PEFT_ID = "Moodoxyy/mistral-7b-dsa-finetuned"

# ============================================================================
# RAG COMPONENTS
# ============================================================================

class RAGRetriever:
    """RAG retriever for enhanced context retrieval."""
    
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        """Initialize RAG retriever with embedding model."""
        print(f"\n🔍 Initializing RAG Retriever with {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunks = []
        self.embeddings = None
        self.index = None
        print("✅ RAG Retriever initialized")
    
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
        
        print(f"✅ Index built with {len(chunks)} chunks")
    
    def retrieve(self, query, top_k=3):
        """Retrieve top-k most relevant chunks for a query."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return relevant chunks
        retrieved_chunks = [self.chunks[idx] for idx in indices[0]]
        return retrieved_chunks, distances[0]
    
    def retrieve_diverse(self, query, top_k=5, diversity_threshold=0.7):
        """Retrieve diverse chunks to cover different aspects."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Get more candidates than needed
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k * 2)
        
        # Select diverse chunks
        selected_chunks = []
        selected_indices = []
        
        for idx in indices[0]:
            if len(selected_chunks) >= top_k:
                break
            
            # Check diversity
            is_diverse = True
            current_chunk = self.chunks[idx]
            
            for selected_chunk in selected_chunks:
                # Simple diversity check based on overlap
                overlap = len(set(current_chunk.split()) & set(selected_chunk.split()))
                similarity = overlap / min(len(current_chunk.split()), len(selected_chunk.split()))
                
                if similarity > diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected_chunks.append(current_chunk)
                selected_indices.append(idx)
        
        return selected_chunks


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load your fine-tuned Mistral model."""
    print("\n" + "="*70)
    print("🧠 Loading Moodoxyy/mistral-7b-dsa-finetuned")
    print("="*70)
    
    print("\n📦 Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    
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
    
    print(f"✅ Fine-tuned model loaded on device: {model.device}")
    print("\n" + "="*70)
    print("🎉 Model ready for MCQ generation!")
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
                page_text = page.extract_text()
                text += page_text + "\n\n"
                
            print(f"✅ Extracted {len(text)} characters")
            
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return None
    
    return text


def chunk_text(text, chunk_size=2000, overlap=200):
    """Split text into overlapping chunks for processing."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
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
# RAG-ENHANCED MCQ GENERATION
# ============================================================================

def create_rag_enhanced_prompt(main_content, related_contexts, difficulty="medium", num_questions=1):
    """Create an enhanced prompt with RAG-retrieved context."""
    
    # Combine related contexts
    context_summary = "\n\n".join([f"Related Context {i+1}:\n{ctx[:500]}..." 
                                   for i, ctx in enumerate(related_contexts)])
    
    prompt = f"""You are an expert at creating high-quality multiple choice questions.

Main Content:
{main_content}

Additional Related Information:
{context_summary}

Task: Generate {num_questions} high-quality multiple choice question(s) based on the main content above.
Use the additional related information to create more comprehensive and contextual questions.

Difficulty: {difficulty}

Requirements:
- Questions should test deep understanding, not just recall
- Options should be plausible and challenging
- Avoid placeholder text or generic questions
- Ensure all 4 options (A, B, C, D) are meaningful and distinct

Format each question EXACTLY as follows:
Question: [specific, clear question text]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
Correct Answer: [A/B/C/D]
Explanation: [detailed explanation of why the answer is correct]

---
"""
    
    return prompt


def generate_mcqs_with_rag(model, tokenizer, retriever, chunk, chunk_idx, 
                           difficulty="medium", num_questions=1, 
                           max_length=1024, temperature=0.7):
    """Generate MCQs using RAG for enhanced context."""
    
    print(f"\n🔍 Retrieving relevant context for chunk {chunk_idx}...")
    
    # Retrieve related chunks using RAG
    related_chunks = retriever.retrieve_diverse(chunk, top_k=3, diversity_threshold=0.6)
    
    print(f"✅ Retrieved {len(related_chunks)} related contexts")
    
    # Create RAG-enhanced prompt
    prompt = create_rag_enhanced_prompt(
        main_content=chunk[:1500],  # Limit main content
        related_contexts=related_chunks,
        difficulty=difficulty,
        num_questions=num_questions
    )
    
    print(f"\n📝 Generating {num_questions} MCQ(s) with RAG enhancement...")
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the output
    mcq_text = generated_text[len(prompt):].strip()
    
    return mcq_text


# ============================================================================
# PARSING & VALIDATION
# ============================================================================

def parse_mcqs(mcq_text):
    """Parse generated MCQ text into structured format."""
    
    questions = []
    question_blocks = mcq_text.split("---")
    
    for block in question_blocks:
        block = block.strip()
        if not block:
            continue
        
        mcq = {}
        
        # Extract question
        q_match = re.search(r'Question:\s*(.+?)(?=\n[A-D]\))', block, re.DOTALL)
        if q_match:
            mcq['question'] = q_match.group(1).strip()
        
        # Extract options
        options = {}
        for letter in ['A', 'B', 'C', 'D']:
            opt_pattern = rf'{letter}\)\s*(.+?)(?=\n[A-D]\)|\nCorrect Answer:|\nAnswer:|\nExplanation:|$)'
            opt_match = re.search(opt_pattern, block, re.DOTALL)
            
            if opt_match:
                option_text = opt_match.group(1).strip()
                option_text = re.sub(r'\n?Answer:\s*[A-D]\).*$', '', option_text, flags=re.DOTALL).strip()
                options[letter] = option_text
        
        mcq['options'] = options
        
        # Extract correct answer
        ans_match = re.search(r'Correct Answer:\s*([A-D])', block, re.IGNORECASE)
        if ans_match:
            mcq['correct_answer'] = ans_match.group(1).strip().upper()
        else:
            ans_match = re.search(r'Answer:\s*([A-D])\)?', block, re.IGNORECASE)
            if ans_match:
                mcq['correct_answer'] = ans_match.group(1).strip().upper()
        
        # Extract explanation
        exp_match = re.search(r'Explanation:\s*(.+?)(?=\n\nQuestion:|\n---|\Z)', block, re.DOTALL | re.IGNORECASE)
        if exp_match:
            mcq['explanation'] = exp_match.group(1).strip()
        
        # Validate basic structure
        if mcq.get('question') and mcq.get('options'):
            if 'correct_answer' not in mcq:
                for letter, option in mcq['options'].items():
                    if re.search(r'Answer:\s*' + letter, block, re.IGNORECASE):
                        mcq['correct_answer'] = letter
                        break
            
            if 'explanation' not in mcq or not mcq['explanation']:
                mcq['explanation'] = "No explanation provided."
            
            questions.append(mcq)
    
    return questions


def clean_and_validate_mcqs(questions):
    """Clean and validate parsed MCQs with comprehensive quality checks."""
    
    cleaned_questions = []
    
    for i, mcq in enumerate(questions, 1):
        cleaned_mcq = {}
        is_valid = True
        
        # VALIDATE QUESTION
        if 'question' not in mcq or not mcq['question']:
            print(f"❌ Question {i}: Missing question text, skipping...")
            continue
        
        question_text = mcq['question'].strip()
        
        # Check for placeholder text
        placeholder_patterns = [
            r'\[.*?\]',
            r'Based on the (above|given) content',
            r'generate (a|another) (multiple choice )?question',
            r'specific, clear question text'
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, question_text, re.IGNORECASE):
                print(f"❌ Question {i}: Contains placeholder text, skipping...")
                is_valid = False
                break
        
        if not is_valid:
            continue
        
        if len(question_text) < 20:
            print(f"❌ Question {i}: Question too short (< 20 chars), skipping...")
            continue
        
        cleaned_mcq['question'] = question_text
        
        # VALIDATE OPTIONS
        if 'options' not in mcq or len(mcq['options']) < 2:
            print(f"❌ Question {i}: Insufficient options, skipping...")
            continue
        
        cleaned_options = {}
        valid_options_count = 0
        
        for letter in ['A', 'B', 'C', 'D']:
            if letter in mcq['options']:
                option_text = mcq['options'][letter].strip()
                option_text = re.sub(r'\s*Answer:.*$', '', option_text, flags=re.IGNORECASE)
                
                # Check for placeholders
                if re.search(r'^\[option [A-D]\]$', option_text, re.IGNORECASE):
                    continue
                
                if len(option_text) < 2:
                    continue
                
                if option_text in cleaned_options.values():
                    continue
                
                cleaned_options[letter] = option_text
                valid_options_count += 1
        
        if valid_options_count < 2:
            print(f"❌ Question {i}: Less than 2 valid options, skipping...")
            continue
        
        cleaned_mcq['options'] = cleaned_options
        
        # VALIDATE ANSWER
        if 'correct_answer' in mcq:
            answer = mcq['correct_answer'].strip().upper()
            if answer in cleaned_mcq['options']:
                cleaned_mcq['correct_answer'] = answer
            else:
                cleaned_mcq['correct_answer'] = list(cleaned_mcq['options'].keys())[0]
        else:
            cleaned_mcq['correct_answer'] = list(cleaned_mcq['options'].keys())[0]
        
        # VALIDATE EXPLANATION
        if 'explanation' in mcq and mcq['explanation']:
            explanation = mcq['explanation'].strip()
            explanation = re.sub(r'\n\nQuestion:.*$', '', explanation, flags=re.DOTALL)
            
            if len(explanation) < 10 or re.search(r'^\[.*?\]$', explanation):
                explanation = f"The correct answer is {cleaned_mcq['correct_answer']}."
            
            cleaned_mcq['explanation'] = explanation
        else:
            cleaned_mcq['explanation'] = f"The correct answer is {cleaned_mcq['correct_answer']}."
        
        cleaned_questions.append(cleaned_mcq)
        print(f"✅ Question {i}: Validated successfully")
    
    return cleaned_questions


# ============================================================================
# OUTPUT & DISPLAY
# ============================================================================

def display_mcqs(questions):
    """Display MCQs in a readable format."""
    
    if not questions:
        print("\n⚠️  No valid questions to display!")
        return
    
    print("\n" + "="*70)
    print("📋 Generated MCQs")
    print("="*70 + "\n")
    
    for i, mcq in enumerate(questions, 1):
        print(f"Question {i}: {mcq.get('question', 'N/A')}")
        print()
        
        for letter, option in mcq.get('options', {}).items():
            print(f"{letter}) {option}")
        print()
        
        print(f"✅ Correct Answer: {mcq.get('correct_answer', 'N/A')}")
        print(f"💡 Explanation: {mcq.get('explanation', 'N/A')}")
        print("\n" + "-"*70 + "\n")


def print_validation_summary(total_parsed, total_valid, total_chunks):
    """Print a summary of the validation process."""
    
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)
    print(f"Total chunks processed: {total_chunks}")
    print(f"Total questions parsed: {total_parsed}")
    print(f"Valid questions after validation: {total_valid}")
    
    if total_parsed > 0:
        validation_rate = (total_valid / total_parsed) * 100
        print(f"Validation success rate: {validation_rate:.1f}%")
        
        rejected = total_parsed - total_valid
        print(f"Rejected questions: {rejected}")
        
        if rejected > 0:
            print(f"\n⚠️  {rejected} question(s) were rejected due to quality issues")
    
    print("="*70 + "\n")


def save_mcqs(questions, filename="generated_mcqs.json"):
    """Save MCQs to a JSON file."""
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    
    print(f"💾 MCQs saved to {filename}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def generate_mcqs_from_pdf_with_rag(model, tokenizer, pdf_path, difficulty="medium", 
                                    questions_per_chunk=2, max_chunks=None, 
                                    output_file=None, use_rag=True):
    """
    Main function to generate MCQs from PDF with RAG enhancement.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        pdf_path: Path to PDF file
        difficulty: Difficulty level
        questions_per_chunk: Questions per chunk
        max_chunks: Max chunks to process (None = all)
        output_file: Output JSON filename
        use_rag: Enable RAG enhancement
    """
    
    print("\n" + "="*70)
    print("🚀 Starting RAG-Enhanced PDF MCQ Generation Pipeline")
    print("="*70)
    
    # Extract text
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        print("❌ Failed to extract text from PDF")
        return []
    
    # Split into chunks
    print(f"\n✂️  Splitting text into chunks...")
    chunks = chunk_text(pdf_text, chunk_size=2000, overlap=200)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Initialize RAG retriever
    retriever = None
    if use_rag:
        retriever = RAGRetriever()
        retriever.build_index(chunks)
    
    # Limit chunks if specified
    if max_chunks:
        chunks = chunks[:max_chunks]
        print(f"📊 Processing {len(chunks)} chunks (limited)")
    else:
        print(f"📊 Processing all {len(chunks)} chunks")
    
    # Generate MCQs
    all_questions = []
    total_parsed = 0
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n{'='*70}")
        print(f"🔄 Processing Chunk {i}/{len(chunks)}")
        print(f"{'='*70}")
        
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        print(f"Preview: {preview}\n")
        
        try:
            if use_rag and retriever:
                mcq_text = generate_mcqs_with_rag(
                    model=model,
                    tokenizer=tokenizer,
                    retriever=retriever,
                    chunk=chunk,
                    chunk_idx=i,
                    difficulty=difficulty,
                    num_questions=questions_per_chunk,
                    max_length=1024,
                    temperature=0.7
                )
            else:
                # Fallback to non-RAG generation
                from create_mcq_prompt import create_mcq_prompt
                prompt = create_mcq_prompt(chunk[:1500], difficulty, questions_per_chunk)
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=1024, temperature=0.7)
                
                mcq_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            questions = parse_mcqs(mcq_text)
            total_parsed += len(questions)
            
            print(f"\n📝 Parsed {len(questions)} question(s), validating...")
            
            questions = clean_and_validate_mcqs(questions)
            all_questions.extend(questions)
            
            print(f"✅ {len(questions)} valid question(s) from this chunk")
            
        except Exception as e:
            print(f"❌ Error generating MCQs: {e}")
            continue
    
    # Print summary
    print_validation_summary(total_parsed, len(all_questions), len(chunks))
    
    # Display and save
    display_mcqs(all_questions)
    
    if output_file is None:
        pdf_name = Path(pdf_path).stem
        output_file = f"{pdf_name}_mcqs_rag.json"
    
    save_mcqs(all_questions, output_file)
    
    print(f"\n✨ Complete! Generated {len(all_questions)} total questions")
    
    return all_questions


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Load model
    model, tokenizer = load_model()
    
    # Generate MCQs with RAG
    pdf_path = "sample1.pdf"  # Replace with your PDF path
    
    questions = generate_mcqs_from_pdf_with_rag(
        model=model,
        tokenizer=tokenizer,
        pdf_path=pdf_path,
        difficulty="medium",
        questions_per_chunk=2,
        max_chunks=None,  # Process all chunks
        output_file="my_mcqs_rag.json",
        use_rag=True  # Enable RAG enhancement
    )
    
    print(f"\n🎉 Successfully generated {len(questions)} MCQs with RAG!")
    
    # To disable RAG (use standard generation):
    # questions = generate_mcqs_from_pdf_with_rag(..., use_rag=False)