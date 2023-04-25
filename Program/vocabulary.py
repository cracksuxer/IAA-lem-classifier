from rich.console import Console
from rich.traceback import install
from collections import OrderedDict

from spellchecker import SpellChecker
import regex as re # type: ignore
from typing import List
import time
import csv

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

spell = SpellChecker()

PUNCTUATION_FILTER = ['.', ',', '!', '?', '\'', '\"', ':', ';', '(', ')',
                      '[', ']', '{', '}', '...', '…', '’', '‘', '”', '“',
                      '—', '–', '´', '´', '°', '•', '·', '…', '‹', '›',
                      '«', '»', '¿', '¡', '-', '_', '—', '%', '$', '#', 
                      '\ufeff', '\t', '\n', '\r', '\v', '\f', '\a', '\b',
                      '\f', '&', '<', '>', '/', '\\', '|', '@', '~', '`'
                      '+', '=', '*', '^', '§', '¶', '•', '©', '®', '™']

install(show_locals=True, theme="monokai")
console = Console()
arepl_filter = ['console', 'Console', 'stop_word', 'PUNCTUATION_FILTER']
filter_tokens = ['\d+', '\W+', '\b\w{1,2}\b']

def parse_text(text: str) -> List[List[str]]:
    lines = text.strip().split('\n')
    result: List[List[str]] = []
    for line in lines:
        if lines == ',':
            return result
        key_value = line.split(',', 1)
        key = key_value[0].strip()
        value = key_value[1].strip()
        result.append([key, value])
    return result

def to_casefold(text: str) -> str:
    return text.casefold()

def remove_punctuation(text: str) -> str:
    for p in PUNCTUATION_FILTER:
        text = text.replace(p, '')
    return text

def filter_word(word: str) -> bool:
    return not any(re.match(pattern, word) for pattern in filter_tokens)

def remove_stop_word(text: str) -> str:
    text_list = text.split(' ')
    stop_words = open('../stop_words_english.txt', 'r', encoding='utf-8').read().split('\n')
    result = [word for word in text_list if word not in stop_words]
    return ' '.join(result)

def remove_url_hashtags(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'www\.[a-zA-Z0-9]+\.[a-zA-Z0-9]+\.[a-zA-Z0-9]+', '', text)
    return text

def correct_spelling(text: str) -> str:
    words = text.split()
    filtered_words = [word for word in words if filter_word(word)]
    corrected_words = [spell.correction(word) for word in filtered_words]
    return ' '.join(word for word in corrected_words if word is not None)

def preprocess_text(text: str) -> str:
    console.print(text, style='bold red')
    text = to_casefold(text)
    text = remove_punctuation(text)
    text = remove_url_hashtags(text)
    text = remove_stop_word(text)
    text = correct_spelling(text)
    text = remove_stop_word(text)
    return text

def truncate(word: str) -> str:
    stemmer = PorterStemmer()
    return stemmer.stem(word) # type: ignore

def lemmatize(word: str) -> str:
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word) # type: ignore

def save_to_csv(tokenized_list: List[List[str]], filename: str) -> None:
    with open(filename, 'w', newline='') as data_file:
        writer = csv.writer(data_file)
        for row in tokenized_list:
            writer.writerow(row)
            
def read_csv(filename: str) -> List[List[str]]:
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        return list(reader)
    
def save_set_to_txt(tokenized_set: set[str], filename: str) -> None:
    with open(filename, 'w') as f:
        for token in tokenized_set:
            f.write(f'{token}, ')
            
def read_txt(filename: str) -> set[str]:
    with open(filename, 'r') as f:
        return set(f.read().split(', '))

def tokenize() -> None:
    body = open('../F75_train.csv', 'r', encoding='utf-8').read()
    result = parse_text(body)
    
    start = time.perf_counter_ns()
    preprocessed_text = [preprocess_text(text[0]) for text in result]
    end = time.perf_counter_ns()
    
    console.print(f'Preprocessing time: {end - start} ns', style='bold blue')
    
    truncate_words, lemmatize_words = [], []
    
    for text in preprocessed_text:
        truncate_words.append([truncate(word) for word in text.split()])
        lemmatize_words.append([lemmatize(word) for word in text.split()])
        
    save_to_csv(truncate_words, 'truncate_words.csv')
    save_to_csv(lemmatize_words, 'lemmatize_words.csv')
    
    truncate_words_set: set[str] = set()
    truncate_words_set.update(*truncate_words)
    
    lemmatize_words_set: set[str] = set()
    lemmatize_words_set.update(*lemmatize_words)
    
    save_set_to_txt(truncate_words_set, 'truncate_words.txt')
    save_set_to_txt(lemmatize_words_set, 'lemmatize_words.txt')
    
def correct_set_words(word_set: set[str]) -> List[str]:
    corrected_words = [spell.correction(word) for word in word_set]
    return [word for word in corrected_words if word and word in spell]

def analyze() -> None:
    truncate_words = read_txt('truncate_words.txt')
    lemmatize_words = read_txt('lemmatize_words.txt')

    remove_incorrect_words(truncate_words, 'truncated_corrected_words.txt')
    remove_incorrect_words(lemmatize_words, 'lemmatize_corrected_words.txt')

def remove_incorrect_words(set_words: set[str], filename: str) -> None:
    corrected_truncated_words = [spell.correction(word) for word in set_words]
    result_trucated: List[str] = [word for word in corrected_truncated_words if word and word in spell]
    save_set_to_txt(set(result_trucated), filename)
    
if __name__ == "__main__":
    tokenize()
    analyze()