import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
import re
import PyPDF2
from pathlib import Path

MODEL_ID = "mistralai/Mistral-7B-v0.1"
PEFT_ID = "Moodoxyy/mistral-7b-dsa-finetuned"

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


def create_mcq_prompt(content, difficulty="medium", num_questions=1):
    """Create a structured prompt for MCQ generation from PDF content."""
    
    prompt = f"""Based on the following content, generate {num_questions} multiple choice question(s).

Content:
{content}

Difficulty: {difficulty}

Format each question as follows:
Question: [question text]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
Correct Answer: [A/B/C/D]
Explanation: [brief explanation]

---
"""
    
    return prompt


def generate_mcqs_from_text(model, tokenizer, content, difficulty="medium", 
                            num_questions=1, max_length=1024, temperature=0.7):
    """Generate MCQs from text content."""
    
    # Truncate content if too long
    max_content_length = 1500
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."
    
    prompt = create_mcq_prompt(content, difficulty, num_questions)
    
    print(f"\n📝 Generating {num_questions} MCQ(s)...")
    
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


def parse_mcqs(mcq_text):
    """Parse generated MCQ text into structured format with robust handling."""
    
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
        
        # Extract options with multiple parsing strategies
        options = {}
        for letter in ['A', 'B', 'C', 'D']:
            opt_pattern = rf'{letter}\)\s*(.+?)(?=\n[A-D]\)|\nCorrect Answer:|\nAnswer:|\nExplanation:|$)'
            opt_match = re.search(opt_pattern, block, re.DOTALL)
            
            if opt_match:
                option_text = opt_match.group(1).strip()
                
                # Clean up: Remove embedded answers from option text
                option_text = re.sub(r'\n?Answer:\s*[A-D]\).*$', '', option_text, flags=re.DOTALL).strip()
                
                options[letter] = option_text
        
        mcq['options'] = options
        
        # Extract correct answer with multiple formats
        ans_match = re.search(r'Correct Answer:\s*([A-D])', block, re.IGNORECASE)
        if ans_match:
            mcq['correct_answer'] = ans_match.group(1).strip().upper()
        else:
            ans_match = re.search(r'Answer:\s*([A-D])\)?', block, re.IGNORECASE)
            if ans_match:
                mcq['correct_answer'] = ans_match.group(1).strip().upper()
        
        # Extract explanation with multiple formats
        exp_match = re.search(r'Explanation:\s*(.+?)(?=\n\nQuestion:|\n---|\Z)', block, re.DOTALL | re.IGNORECASE)
        if exp_match:
            mcq['explanation'] = exp_match.group(1).strip()
        else:
            exp_match = re.search(r'Answer:\s*[A-D]\)\s*[^.]+\.\s*(.+?)(?=\n\nQuestion:|\n---|\Z)', block, re.DOTALL)
            if exp_match:
                mcq['explanation'] = exp_match.group(1).strip()
        
        # Validate and clean up the MCQ
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
    """Clean and validate parsed MCQs to ensure consistency and quality."""
    
    cleaned_questions = []
    
    for i, mcq in enumerate(questions, 1):
        cleaned_mcq = {}
        is_valid = True
        
        # VALIDATE QUESTION TEXT
        if 'question' not in mcq or not mcq['question']:
            print(f"❌ Question {i}: Missing question text, skipping...")
            continue
        
        question_text = mcq['question'].strip()
        
        # Check for placeholder text
        placeholder_patterns = [
            r'\[.*?\]',
            r'Based on the (above|given) content',
            r'generate (a|another) (multiple choice )?question'
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, question_text, re.IGNORECASE):
                print(f"❌ Question {i}: Contains placeholder text, skipping...")
                is_valid = False
                break
        
        if not is_valid:
            continue
        
        # Check minimum question length
        if len(question_text) < 20:
            print(f"❌ Question {i}: Question too short (< 20 chars), skipping...")
            continue
        
        cleaned_mcq['question'] = question_text
        
        # VALIDATE OPTIONS
        if 'options' not in mcq or len(mcq['options']) < 2:
            print(f"❌ Question {i}: Insufficient options (< 2), skipping...")
            continue
        
        cleaned_options = {}
        valid_options_count = 0
        
        for letter in ['A', 'B', 'C', 'D']:
            if letter in mcq['options']:
                option_text = mcq['options'][letter].strip()
                
                # Remove any trailing answer indicators
                option_text = re.sub(r'\s*Answer:.*$', '', option_text, flags=re.IGNORECASE)
                
                # Check for placeholder options
                if re.search(r'^\[.*?\]$', option_text):
                    print(f"⚠️  Question {i}: Option {letter} is placeholder, skipping this option...")
                    continue
                
                # Check for empty or very short options
                if len(option_text) < 2:
                    print(f"⚠️  Question {i}: Option {letter} too short, skipping this option...")
                    continue
                
                # Check for duplicate options
                if option_text in cleaned_options.values():
                    print(f"⚠️  Question {i}: Option {letter} is duplicate, skipping this option...")
                    continue
                
                cleaned_options[letter] = option_text
                valid_options_count += 1
        
        # Need at least 2 valid options
        if valid_options_count < 2:
            print(f"❌ Question {i}: Less than 2 valid options, skipping...")
            continue
        
        cleaned_mcq['options'] = cleaned_options
        
        # VALIDATE CORRECT ANSWER
        if 'correct_answer' in mcq:
            answer = mcq['correct_answer'].strip().upper()
            if answer in cleaned_mcq['options']:
                cleaned_mcq['correct_answer'] = answer
            else:
                print(f"⚠️  Question {i}: Invalid answer '{answer}', using first available option")
                cleaned_mcq['correct_answer'] = list(cleaned_mcq['options'].keys())[0]
        else:
            print(f"⚠️  Question {i}: Missing correct answer, using first available option")
            cleaned_mcq['correct_answer'] = list(cleaned_mcq['options'].keys())[0]
        
        # VALIDATE EXPLANATION
        if 'explanation' in mcq and mcq['explanation']:
            explanation = mcq['explanation'].strip()
            
            # Remove any extra question/answer fragments
            explanation = re.sub(r'\n\nQuestion:.*$', '', explanation, flags=re.DOTALL)
            
            # Check for placeholder explanations
            if re.search(r'^\[.*?\]$', explanation):
                explanation = f"The correct answer is {cleaned_mcq['correct_answer']}."
            
            # Check for incomplete explanations
            if len(explanation) < 10:
                explanation = f"The correct answer is {cleaned_mcq['correct_answer']}."
            
            cleaned_mcq['explanation'] = explanation
        else:
            cleaned_mcq['explanation'] = f"The correct answer is {cleaned_mcq['correct_answer']}."
        
        # FINAL QUALITY CHECK
        question_words = set(question_text.lower().split())
        option_overlap = 0
        
        for option in cleaned_mcq['options'].values():
            option_words = set(option.lower().split())
            overlap = len(question_words & option_words)
            if overlap > len(question_words) * 0.7:
                option_overlap += 1
        
        if option_overlap >= 3:
            print(f"⚠️  Question {i}: High overlap between question and options, but keeping...")
        
        # Add the validated question
        cleaned_questions.append(cleaned_mcq)
        print(f"✅ Question {i}: Validated successfully")
    
    return cleaned_questions


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
            print(f"\n⚠️  {rejected} question(s) were rejected due to:")
            print("   - Placeholder text ([Question], [Option A], etc.)")
            print("   - Insufficient content length")
            print("   - Invalid or missing options")
            print("   - Duplicate options")
    
    print("="*70 + "\n")


def save_mcqs(questions, filename="generated_mcqs.json"):
    """Save MCQs to a JSON file."""
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    
    print(f"💾 MCQs saved to {filename}")


def generate_mcqs_from_pdf(model, tokenizer, pdf_path, difficulty="medium", 
                           questions_per_chunk=2, max_chunks=None, output_file=None):
    """
    Main function to generate MCQs from a PDF file.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        pdf_path: Path to PDF file
        difficulty: Difficulty level (easy/medium/hard)
        questions_per_chunk: Number of questions to generate per text chunk
        max_chunks: Maximum number of chunks to process (None = all chunks)
        output_file: Output JSON filename
    """
    
    print("\n" + "="*70)
    print("🚀 Starting PDF MCQ Generation Pipeline")
    print("="*70)
    
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        print("❌ Failed to extract text from PDF")
        return []
    
    # Split into chunks
    print(f"\n✂️  Splitting text into chunks...")
    chunks = chunk_text(pdf_text, chunk_size=2000, overlap=200)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Limit number of chunks to process
    if max_chunks:
        chunks = chunks[:max_chunks]
        print(f"📊 Processing {len(chunks)} chunks (limited by max_chunks)")
    else:
        print(f"📊 Processing all {len(chunks)} chunks")
    
    # Generate MCQs from each chunk
    all_questions = []
    total_parsed = 0
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n{'='*70}")
        print(f"🔄 Processing Chunk {i}/{len(chunks)}")
        print(f"{'='*70}")
        
        # Show preview of chunk
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        print(f"Preview: {preview}\n")
        
        try:
            mcq_text = generate_mcqs_from_text(
                model=model,
                tokenizer=tokenizer,
                content=chunk,
                difficulty=difficulty,
                num_questions=questions_per_chunk,
                max_length=1024,
                temperature=0.7
            )
            
            questions = parse_mcqs(mcq_text)
            total_parsed += len(questions)
            
            print(f"\n📝 Parsed {len(questions)} question(s), now validating...")
            
            # Clean and validate
            questions = clean_and_validate_mcqs(questions)
            
            all_questions.extend(questions)
            
            print(f"✅ {len(questions)} valid question(s) from this chunk")
            
        except Exception as e:
            print(f"❌ Error generating MCQs: {e}")
            continue
    
    # Print validation summary
    print_validation_summary(total_parsed, len(all_questions), len(chunks))
    
    # Display all generated MCQs
    display_mcqs(all_questions)
    
    # Save to file
    if output_file is None:
        pdf_name = Path(pdf_path).stem
        output_file = f"{pdf_name}_mcqs.json"
    
    save_mcqs(all_questions, output_file)
    
    print(f"\n✨ Complete! Generated {len(all_questions)} total questions")
    
    return all_questions


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Load model
    model, tokenizer = load_model()
    
    # Example: Generate MCQs from a PDF (processes ALL chunks)
    pdf_path = "sample.pdf"  # Replace with your PDF path
    
    questions = generate_mcqs_from_pdf(
        model=model,
        tokenizer=tokenizer,
        pdf_path=pdf_path,
        difficulty="medium",
        questions_per_chunk=2,
        max_chunks=None,  # Process ALL chunks
        output_file="my_mcqs.json"
    )
    
    print(f"\n🎉 Successfully generated {len(questions)} MCQs from PDF!")
    
    # Optional: Limit chunks for testing
    # questions = generate_mcqs_from_pdf(
    #     model=model,
    #     tokenizer=tokenizer,
    #     pdf_path=pdf_path,
    #     max_chunks=5  # Only process first 5 chunks
    # )