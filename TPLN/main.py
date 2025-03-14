import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import PyPDF2
from bs4 import BeautifulSoup
import re
import numpy as np
import torch

# ====== Pentru rezumare EXTRACTIVĂ (Sentence-BERT) ======
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ====== Pentru rezumare ABSTRACTIVĂ (mT5) ======
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------------
# 0. SETĂM DEVICE-UL (GPU SAU CPU)
# --------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Folosește device:", device)

# --------------------------------------------------------
# 1. MODELE & RESURSE
# --------------------------------------------------------

# 1.1 spaCy pentru limba română
nlp = spacy.load("ro_core_news_lg")

# 1.2 Sentence-BERT (multilingv) pentru rezumare EXTRACTIVĂ
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

# 1.3 mT5 pentru rezumare ABSTRACTIVĂ (iliemihai/mt5-base-romanian-diacritics)
abstractive_model_name = "iliemihai/mt5-base-romanian-diacritics"
abstractive_tokenizer = AutoTokenizer.from_pretrained(abstractive_model_name, use_fast=False)
abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(abstractive_model_name).to(device)

# --------------------------------------------------------
# 2. PREPROCESARE TEXT (PDF, HTML, etc.)
# --------------------------------------------------------

def extract_text_from_pdf(pdf_path):
    """
    Extrage textul dintr-un fișier PDF folosind PyPDF2.
    """
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_html(html_content):
    """
    Extrage textul dintr-un conținut HTML folosind BeautifulSoup.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=' ')

def clean_text(text):
    """
    Curăță textul: elimină spații multiple și caractere nedorite,
    păstrând doar litere, cifre și semne de punctuație de bază.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-ZÀ-ɏḀ-ỿ0-9 .,!?\n]', '', text)
    return text.strip()

def tokenize_text(text):
    """
    Împarte textul în propoziții (nltk.sent_tokenize),
    apoi fiecare propoziție în cuvinte (nltk.word_tokenize).
    Returnează (sentences, words).
    """
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]
    return sentences, words

def preprocess_text_spacy(text):
    """
    Opțional: folosind spaCy pentru a elimina stopwords și punctuație.
    """
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# --------------------------------------------------------
# 3. REZUMARE EXTRACTIVĂ (Sentence-BERT)
# --------------------------------------------------------

def get_sentence_embeddings(sentences):
    """
    Transformă fiecare propoziție într-un embedding cu Sentence-BERT.
    """
    return embedding_model.encode(sentences, convert_to_numpy=True)

def extractive_summarize(sentences, top_n=5):
    """
    Calculează un scor pentru fiecare propoziție (suma similarităților
    cosinus cu celelalte propoziții) și alege top_n propoziții.
    """
    if not sentences:
        return []

    embeddings = get_sentence_embeddings(sentences)
    sim_matrix = cosine_similarity(embeddings)
    scores = np.sum(sim_matrix, axis=1)

    top_indices = np.argsort(scores)[-top_n:][::-1]
    summary_sentences = [sentences[i] for i in top_indices]
    return summary_sentences


# --------------------------------------------------------
# 5. REZUMARE ABSTRACTIVĂ (mT5 - Sampling, Mai Lung)
# --------------------------------------------------------

def abstractive_summarize(text):
    """
    Generează un rezumat abstractive în română, foarte lung,
    folosind sampling (nu beam search). Astfel, textul final e mai
    creativ și nu copiază direct propozițiile.

    length_penalty=0.3 => modelează preferința pentru texte lungi.
    top_k, top_p => sampling.
    temperature => >1 => text mai inventiv.
    """

    # Prompt explicit
    prompt_text = (
            "Te rog să creezi un rezumat cât mai lung, coerent și original, "
            "fără a copia direct frazele, pentru textul următor: "
            + text
    )

    input_ids = abstractive_tokenizer.encode(
        prompt_text,
        return_tensors="pt",
        truncation=False  # Atenție: dacă textul e enorm, poate genera eroare
    ).to(device)

    # Generare cu sampling
    summary_ids = abstractive_model.generate(
        input_ids,
        max_length=3100,
        min_length=300,       # nu forța să fie prea lung
        do_sample=True,
        temperature=0.5,      # sub 1 => mai puțin creativ
        top_p=0.6,           # restrânge aria sampling-ului
        top_k=50,             # se iau în considerare primii 50 de tokeni
        repetition_penalty=1.1,  # penalizează repetarea
        no_repeat_ngram_size=3,
        early_stopping=False  # poți lăsa așa
    )

    summary = abstractive_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# --------------------------------------------------------
# 6. DEMO DE UTILIZARE
# --------------------------------------------------------

if __name__ == "__main__":
    # 1) Citim textul din PDF
    pdf_text = extract_text_from_pdf("stefan cel mare.pdf")  # Înlocuiește cu calea ta
    cleaned_pdf_text = clean_text(pdf_text)
    pdf_sentences, pdf_words = tokenize_text(cleaned_pdf_text)

    # 2) REZUMARE EXTRACTIVĂ
    extractive_summary_pdf = extractive_summarize(pdf_sentences, top_n=5)
    print("Rezumat EXTRACTIV (PDF):")
    for sentence in extractive_summary_pdf:
        print(" -", sentence)

    # 3) REZUMARE ABSTRACTIVĂ (direct)
    concatenated_pdf_text = " ".join(pdf_sentences)
    abstractive_summary_pdf = abstractive_summarize(concatenated_pdf_text)
    print("\nRezumat ABSTRACTIV (PDF) - direct (sampling):")
    print(abstractive_summary_pdf)

