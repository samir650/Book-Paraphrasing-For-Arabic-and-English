from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from langdetect import detect
from bidi.algorithm import get_display
import pdfplumber
import arabic_reshaper
from fpdf import FPDF
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import stanza
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

# Configure upload and paraphrased directories
UPLOAD_FOLDER = 'uploads'
PARAPHRASED_FOLDER = 'paraphrased'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PARAPHRASED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PARAPHRASED_FOLDER'] = PARAPHRASED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

# Allowed extensions
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Download NLTK data
nltk.download('punkt')

# Download and initialize Stanza pipeline for Arabic
stanza.download('ar')
nlp_ar = stanza.Pipeline('ar', processors='tokenize')

# Load models at startup
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=0)  # Change to 'cuda' if GPU available
paraphraser_en = pipeline("text2text-generation", model="t5-base", device=0)  # Change device as needed

model_name = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
paraphraser_ar = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)  # Change device as needed

# Initialize DejaVu font paths
FONT_PATH = os.path.join('fonts', 'DejaVuSans.ttf')  # Ensure the font file is available
BOLD_FONT_PATH = os.path.join('fonts', 'DejaVuSans-Bold.ttf')  # Ensure the bold font file is available

# Ensure the font files exist
if not os.path.exists(FONT_PATH) or not os.path.exists(BOLD_FONT_PATH):
    raise FileNotFoundError("Font files not found in the 'fonts/' directory.")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''.join(page.extract_text() for page in pdf.pages)
    return text

# Arabic text reshaping and bidi fix
def fix_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# Clean text for paraphrasing
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\b[A-Za-z]\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean Arabic text
def clean_arabic_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\b[ء-ي]\b', '', text)
    text = re.sub(r'[?!؟"()«»:\-]', '', text)
    text = re.sub(r'\s+([,.،])', r'\1', text)
    text = re.sub(r'([,.،])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean and paraphrase all chunks
def clean_chunks(chunks):
    return [clean_arabic_text(chunk) for chunk in chunks]

# English chunking based on semantic similarity
def divide_by_semantics_with_length(text, threshold=0.6, max_words=150, min_words=100):
    sentences = nltk.sent_tokenize(text)  # Use NLTK for sentence tokenization
    embeddings = sbert_model.encode(sentences, convert_to_tensor=True)
    chunks = []
    current_chunk = sentences[0]

    for i in range(1, len(sentences)):
        similarity = util.pytorch_cos_sim(embeddings[i], embeddings[i-1])
        current_word_count = len(current_chunk.split())

        # If the next sentence makes the chunk exceed the max word limit
        if current_word_count + len(sentences[i].split()) > max_words:
            # Ensure the current chunk has at least min_words before breaking
            if current_word_count >= min_words:
                chunks.append(current_chunk.strip())  # Finalize the current chunk
                current_chunk = sentences[i]  # Start a new chunk
            else:
                # If the chunk is below min_words, add the sentence even if it exceeds max_words
                current_chunk += ' ' + sentences[i]
        elif similarity < threshold:
            # Break chunk if semantic similarity is low and the chunk meets the minimum word count
            if current_word_count >= min_words:
                chunks.append(current_chunk.strip())  # Finalize the current chunk
                current_chunk = sentences[i]  # Start a new chunk
            else:
                # If the chunk is too small, continue adding sentences
                current_chunk += ' ' + sentences[i]
        else:
            # Continue adding sentences to the current chunk
            current_chunk += ' ' + sentences[i]

    # Append the last chunk if it satisfies the minimum word condition
    if len(current_chunk.split()) >= min_words:
        chunks.append(current_chunk.strip())

    return chunks

# Arabic chunking function
def chunk_arabic_text(text, tokenizer, max_tokens=300):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ''
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))

        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                # إذا كانت الجملة نفسها تتجاوز الحد الأقصى، نقسمها إلى كلمات
                words = sentence.split()
                sub_chunk = ''
                sub_tokens = 0
                for word in words:
                    word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
                    if sub_tokens + word_tokens > max_tokens:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                            sub_chunk = word
                            sub_tokens = word_tokens
                        else:
                            sub_chunk = ''
                            sub_tokens = 0
                    else:
                        sub_chunk += ' ' + word
                        sub_tokens += word_tokens
                if sub_chunk:
                    chunks.append(sub_chunk.strip())
        else:
            current_chunk += ' ' + sentence
            current_tokens += sentence_tokens

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# Paraphrasing functions
def paraphrase_chunks_en(chunks, min_words=100, max_words=150, num_return_sequences=1):
    paraphrased_chunks = []
    for chunk in chunks:
        chunk_length = len(chunk.split())  # Get the word count of the original chunk

        try:
            # Use the paraphraser to generate paraphrases
            paraphrases = paraphraser_en(chunk, max_length=max_words, min_words=min_words, num_return_sequences=num_return_sequences, do_sample=False)
            paraphrased_text = paraphrases[0]['generated_text']  # Extract the paraphrased text

            paraphrased_chunks.append(paraphrased_text)
        except Exception as e:
            # Log the error for debugging
            print(f"Error paraphrasing chunk: {e}")
            paraphrased_chunks.append(chunk)  # Append the original chunk if paraphrasing fails

    return paraphrased_chunks


def paraphrase_chunks_ar(chunks, min_words=200, max_words=300, num_return_sequences=1):
    paraphrased_chunks = []
    for chunk in chunks:
        chunk_length = len(chunk.split())  # Get the word count of the original chunk

        try:
            paraphrases = paraphraser_ar(chunk, max_length=max_words, min_words=min_words ,num_return_sequences=num_return_sequences, do_sample=False)
            paraphrased_text = paraphrases[0]['generated_text']  # Extract the paraphrased text

            paraphrased_chunks.append(paraphrased_text)
        except Exception as e:
            paraphrased_chunks.append(chunk)  # Append the original chunk if paraphrasing fails

    return paraphrased_chunks

# PDF Generation Setup
reshaped_text = arabic_reshaper.reshape("إعادة صياغة الكتاب")
_text = get_display(reshaped_text)

class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            self.set_font('DejaVu', 'B', 12)
            title = _text if getattr(self, 'language', 'en') == 'ar' else 'Book Paraphrase'
            self.cell(0, 10, title, ln=True, align='C')
            self.ln(10)

    def chapter_body(self, body):
        self.set_font('DejaVu', '', 12)  # Ensure this font supports Arabic
        align = 'R' if getattr(self, 'language', 'ar') == 'ar' else 'L'
        self.multi_cell(0, 10, body, align=align)
        self.ln()

    def add_text(self, text):
        self.add_page()
        self.chapter_body(text)

def generate_pdf(paraphrased_text, pdf_output_path, language='en'):
    pdf = PDF()
    pdf.language = language

    # Load the DejaVu font with Unicode support
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(BASE_DIR, 'fonts')

    pdf.add_font('DejaVu', '', os.path.join(fonts_dir, 'DejaVuSans.ttf'), uni=True)
    pdf.add_font('DejaVu', 'B', os.path.join(fonts_dir, 'DejaVuSans-Bold.ttf'), uni=True)
    pdf.set_font('DejaVu', '', 12)

    pdf.add_text(paraphrased_text)
    pdf.output(pdf_output_path)

def paraphrase_english(book_text, pdf_output_path="english_paraphrase.pdf"):
    # Step 1: Divide text into semantic chunks
    semantic_chunks = divide_by_semantics_with_length(book_text)

    # Step 2: Clean the chunks
    cleaned_chunks = [clean_text(chunk) for chunk in semantic_chunks]

    # Step 3: Paraphrase the chunks
    paraphrased_chunks = paraphrase_chunks_en(cleaned_chunks)

    # Step 4: Generate PDF
    final_paraphrase = '\n\n'.join(paraphrased_chunks)
    generate_pdf(final_paraphrase, pdf_output_path, language='en')

    print(f"Paraphrasing completed! Saved to {pdf_output_path}")

    return final_paraphrase

def paraphrase_arabic(pdf_path, pdf_output_path="arabic_paraphrase.pdf"):
    # Step 1: Extract text from PDF and fix Arabic text direction
    text = extract_text_from_pdf(pdf_path)
    fixed_text = fix_arabic_text(text)  # Fixing the text direction

    # Step 2: Chunk the text semantically
    chunks = chunk_arabic_text(fixed_text, tokenizer, max_tokens=300)  # Now the chunking function is defined

    # Step 3: Paraphrase the chunks
    paraphrased_chunks = paraphrase_chunks_ar(chunks)

    # Step 4: Clean the paraphrased chunks using the custom Arabic cleaning function
    cleaned_paraphrase = clean_chunks(paraphrased_chunks)

    # Step 5: Join the cleaned chunks into the final paraphrased text
    final_paraphrase = '\n\n'.join(cleaned_paraphrase)

    # Step 6: Fix the Arabic text direction before generating the PDF
    final_paraphrase_arabic = fix_arabic_text(final_paraphrase)
    generate_pdf(final_paraphrase_arabic, pdf_output_path, language='ar')

    # Notify the user that the PDF has been created
    print(f"Paraphrasing completed! Saved to {pdf_output_path}")

    return final_paraphrase_arabic

def detect_language_and_paraphrase(pdf_path, pdf_output_path_ar, pdf_output_path_en):
    text = extract_text_from_pdf(pdf_path)
    language = detect(text)
    print(f"Detected language: {language}")

    if language == 'ar':
        print("Detected Arabic. Running Arabic paraphrasing pipeline...")
        return paraphrase_arabic(pdf_path, pdf_output_path=pdf_output_path_ar)
    else:
        print("Detected English. Running English paraphrasing pipeline...")
        return paraphrase_english(text, pdf_output_path=pdf_output_path_en)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'pdf_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['pdf_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            flash('File successfully uploaded')

            try:
                # Determine output file name based on language
                base_name = os.path.splitext(filename)[0]
                text = extract_text_from_pdf(input_path)
                language = detect(text)
                if language == 'ar':
                    output_filename = f"{base_name}_paraphrased_ar.pdf"
                    output_path = os.path.join(app.config['PARAPHRASED_FOLDER'], output_filename)
                    paraphrase_arabic(input_path, output_path)
                else:
                    output_filename = f"{base_name}_paraphrased_en.pdf"
                    output_path = os.path.join(app.config['PARAPHRASED_FOLDER'], output_filename)
                    paraphrase_english(text, output_path)
                
                # **Corrected send_from_directory Call**
                return send_from_directory(app.config['PARAPHRASED_FOLDER'], output_filename, as_attachment=True)
                # Alternatively, if your Flask version requires 'path' keyword:
                # return send_from_directory(directory=app.config['PARAPHRASED_FOLDER'], path=output_filename, as_attachment=True)
            except Exception as e:
                flash(f"An error occurred during paraphrasing: {e}")
                return redirect(request.url)
        else:
            flash('Allowed file types are pdf')
            return redirect(request.url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
