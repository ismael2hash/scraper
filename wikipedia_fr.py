import requests
from bs4 import BeautifulSoup
import re
from langdetect import detect
import time
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import nltk
from nltk.translate.bleu_score import sentence_bleu

nltk.download("punkt")


def get_wikipedia_sentences(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    sentences = []
    for paragraph in paragraphs:
        text = paragraph.get_text()
        # Use regex to split text into sentences
        sentences.extend(re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text))
    return sentences


def is_english(sentence):
    try:
        return detect(sentence) == "en"
    except:
        return False


def preprocess_sentence(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    # Lowercase all tokens
    tokens = [token.lower() for token in tokens]
    # Remove punctuation
    tokens = [token for token in tokens if token.isalnum()]
    return " ".join(tokens)


def collect_10m_sentences():
    start_time = time.time()
    base_url = "https://en.wikipedia.org/wiki/Special:Random"
    total_sentences = 0
    generated_sentences = []
    with open("sentences.txt", "a", encoding="utf-8") as file:
        pbar = tqdm(total=10000000)
        while total_sentences < 10000000:
            sentences = get_wikipedia_sentences(base_url)
            for sentence in sentences:
                if is_english(sentence):
                    preprocessed_sentence = preprocess_sentence(sentence)
                    generated_sentences.append(preprocessed_sentence)
                    file.write(preprocessed_sentence + "\n")
                    total_sentences += 1
                    pbar.update(1)
                    if total_sentences >= 10000000:
                        break
            if total_sentences >= 10000000:
                break
            end_time = time.time()
            elapsed_time = end_time - start_time
            avg_time_per_sentence = (
                elapsed_time / total_sentences if total_sentences > 0 else 0
            )
            remaining_sentences = 10000000 - total_sentences
            estimated_remaining_time = remaining_sentences * avg_time_per_sentence
            print(f"Estimated time remaining: {estimated_remaining_time:.2f} seconds")
    pbar.close()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Finished collecting 10 million sentences in {duration:.2f} seconds.")

    # Calculate BLEU score
    reference = generated_sentences[:100]  # Assuming you have reference sentences
    score = 0
    for sentence in generated_sentences[:100]:
        score += sentence_bleu([reference], sentence)
    avg_bleu_score = score / len(generated_sentences[:100])
    print(f"Average BLEU Score: {avg_bleu_score}")


collect_10m_sentences()
