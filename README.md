# Book Paraphrasing For English and Arabic Pipeline

A comprehensive solution for paraphrasing PDF documents in both English and Arabic. This project leverages advanced NLP techniques to extract, clean, paraphrase, and generate paraphrased PDFs. Additionally, it provides a user-friendly API for seamless integration and usage through a web interface.

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Jupyter Notebook](#running-the-jupyter-notebook)
  - [Using the Web API](#using-the-web-api)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **PDF Text Extraction:** Efficiently extracts text from PDF files using `pdfplumber`.
- **Language Detection:** Automatically identifies whether the PDF content is in English or Arabic.
- **Text Cleaning:** Removes unwanted elements such as URLs, numbers, and extra spaces to ensure clean text.
- **Semantic Chunking:** Divides text into meaningful chunks based on semantic similarity for effective paraphrasing.
- **Paraphrasing:**
  - **English:** Utilizes `T5` and `Sentence-BERT` models.
  - **Arabic:** Employs `mT5` and specialized NLP tools tailored for Arabic text.
- **PDF Generation:** Produces paraphrased PDFs with proper text alignment and formatting, supporting both left-to-right and right-to-left languages.
- **Web Interface:** Offers a simple web interface for uploading PDFs and downloading paraphrased versions.
- **API Integration:** Provides an API endpoint for integrating paraphrasing functionality into other applications or services.

## Folder Structure

```
.
├── Api Code
│   ├── app.py
│   ├── templates
│   │   └── index.html
│   ├── static
│   │   └── styles.css
│   ├── fonts
│   │   ├── DejaVuSans-Bold.ttf
│   │   └── DejaVuSans.ttf
│   ├── uploads
│   └── paraphrased
├── requirements.txt
├── README.md
└── Book_paraphrasing_pipeline_for_english_and_arabic.ipynb
```

### Description of Each Folder/File

- **Api Code/**
  - **app.py:** The Flask application that handles PDF uploads, processing, and returns paraphrased PDFs.
  - **requirements.txt:** Lists all the Python dependencies required to run the API.
  - **templates/**
    - **index.html:** HTML template for the web interface, allowing users to upload PDFs and download paraphrased versions.
  - **static/**
    - **styles.css:** CSS styles for the web interface to ensure a clean and user-friendly design.
  - **fonts/**
    - **DejaVuSans.ttf & DejaVuSans-Bold.ttf:** Font files used by `FPDF` to support Unicode characters, essential for rendering Arabic text correctly.
  - **uploads/:** Directory where the original uploaded PDF files are stored temporarily.
  - **paraphrased/:** Directory where the paraphrased PDF files are saved for download.

- **README.md:** This file. Provides an overview and instructions for the project.

- **[Your Notebook File].ipynb:** The main Jupyter notebook containing the paraphrasing pipeline code, including text extraction, cleaning, paraphrasing, and PDF generation.

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **pip** (Python package installer)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/pdf-paraphrasing-pipeline.git
   cd pdf-paraphrasing-pipeline
   ```

2. **Set Up a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   Navigate to the `Api Code` directory and install the required packages:

   ```bash
   cd "Api Code"
   pip install -r requirements.txt
   ```

4. **Download Font Files**

   Ensure that the `fonts` directory contains `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf`. If not, download them from [DejaVu Fonts](https://dejavu-fonts.github.io/) and place them in the `fonts` folder.

5. **Prepare Uploads and Paraphrased Directories**

   Ensure that the `uploads` and `paraphrased` directories exist. If not, create them:

   ```bash
   mkdir uploads paraphrased
   ```

## Usage

### Running the Jupyter Notebook

1. **Open the Jupyter Notebook**

   Navigate to the project root directory and open the notebook:

   ```bash
   jupyter notebook
   ```

2. **Execute the Notebook Cells**

   Follow the steps outlined in the notebook to install additional libraries, import necessary modules, and run the paraphrasing pipeline on your PDF files.

### Using the Web API

1. **Navigate to the API Code Directory**

   ```bash
   cd "Api Code"
   ```

2. **Run the Flask Application**

   ```bash
   python app.py
   ```

3. **Access the Web Interface**

   Open your browser and navigate to `http://127.0.0.1:5000` to access the web interface. From here, you can upload PDFs and download their paraphrased versions.

## Dependencies

All necessary dependencies are listed in the `Api Code/requirements.txt` file. Key dependencies include:

- **PDF Processing:** `PyMuPDF`, `pdfplumber`, `PyPDF2`
- **NLP and Text Processing:** `nltk`, `spacy`, `stanza`, `transformers`, `sentence-transformers`, `langdetect`, `arabic_reshaper`, `python-bidi`
- **PDF Generation:** `fpdf2`, `reportlab`
- **Web Framework:** `Flask`, `werkzeug`
- **Others:** `matplotlib`, `camel-tools`, `datasets`

### Installing Dependencies

Ensure you have activated your virtual environment (if using one) and run:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Sentence Transformers](https://www.sbert.net/)
- [Stanza NLP](https://stanfordnlp.github.io/stanza/)
- [FPDF Library](https://pyfpdf.readthedocs.io/)
- [DejaVu Fonts](https://dejavu-fonts.github.io/)
- [Flask Framework](https://flask.palletsprojects.com/)

---

By following this guide, you can set up and utilize the PDF Paraphrasing Pipeline effectively. Whether you prefer interacting through a Jupyter notebook or a user-friendly web interface, this project offers flexible options to suit your needs.