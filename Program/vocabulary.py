from rich.console import Console
from rich.traceback import install

import regex as re # type: ignore
from typing import List, Dict
import csv
import string

from nltk.stem import PorterStemmer # type: ignore
from nltk.stem import WordNetLemmatizer
import enchant # type: ignore

install(show_locals=True, theme="monokai")
console = Console()
arepl_filter = ['console', 'Console', 'stop_word', 'PUNCTUATION_FILTER']
filter_tokens = ['\d+', '\W+', '\b\w{1,2}\b']

def parse_text(text: str) -> List[List[str]]:
    lines = text.strip().split('\n')
    result: List[List[str]] = []
    for line in lines:
        if line == ',':
            return result
        key_value = line.split(',')
        key: str = ''
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = key_value[1].strip()
        else:
            for i in range(len(key_value) - 2):
                key_value[0] = f',{key_value[i + 1]}'
            key = key_value[0]
            value = key_value[-1].strip()
        result.append([key, value])
    return result

def to_casefold(text: str) -> str:
    text_list = text.split(' ')
    new_list: List[str] = []
    for word in text_list:
        new_word = word.casefold()
        new_list.append(new_word)
        
    return ' '.join(new_list)

def remove_punctuation(text: str) -> str:
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '').replace('\ufffd', '')
    return text

def filter_word(word: str) -> bool:
    return not any(re.match(pattern, word) for pattern in filter_tokens)

def remove_stop_word(text: str) -> str:
    text_list = text.split(' ')
    stop_words = open('./stop_words_english.txt', 'r', encoding='utf-8').read().split('\n')
    result = [word for word in text_list if word not in stop_words]
    return ' '.join(result)

def remove_url_hashtags(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'www\.[a-zA-Z0-9]+\.[a-zA-Z0-9]+\.[a-zA-Z0-9]+', '', text)
    return text

def correct_spelling(text: str) -> str:
    text = re.sub(r'\d+', '', text)
    words = text.split()
    spell_checker = enchant.Dict("en_US")
    corrected_words = []
    suggestions_cache: Dict[str, List[str]] = {}
    checked_words = set()

    for word in words:
        if word in checked_words:
            corrected_words.append(word)
            continue

        if not spell_checker.check(word):
            if word in suggestions_cache:
                corrected_word = suggestions_cache[word][0]
            elif suggestions := spell_checker.suggest(word):
                corrected_word = suggestions[0]
                suggestions_cache[word] = suggestions
            else:
                corrected_word = word
            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)

        checked_words.add(word)
    return ' '.join(corrected_words)

def remove_one_char_words(text: str) -> str:
    words = text.split()
    new_words: List[str] = [word for word in words if len(word) > 1]
    return " ".join(new_words)

def preprocess_text(text: str) -> str:
    text = to_casefold(text)
    text = remove_punctuation(text)
    text = remove_url_hashtags(text)
    text = remove_stop_word(text)
    text = remove_one_char_words(text)
    text = correct_spelling(text)
    text = remove_punctuation(text)
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
            
def save_to_txt(tokenized: List[str], filename: str, delimiter: str = ', ') -> None:
    with open(filename, 'w') as f:
        for token in tokenized:
            f.write(f'{token}{delimiter}')
    
def read_txt(filename: str, delimiter: str) -> List[str]:
    with open(filename, 'r') as f:
        return list(f.read().split(delimiter))
    
def tokenize() -> None:
    body = open('./train.csv', 'r', encoding='utf-8').read()
    result = parse_text(body)

    preprocessed_text = [preprocess_text(text[0]) for text in result]
    lemmatize_words = [list(text.split()) for text in preprocessed_text]
    save_to_csv(lemmatize_words, 'lemmatize_words_2.csv')

    lemmatize_words_set: set[str] = set()
    lemmatize_words_set.update(*lemmatize_words)

    aux_lem_ordered_list: List[str] = sorted(lemmatize_words_set)

    save_to_txt(aux_lem_ordered_list, './Vocabulary/vocabulary_2.txt', '\n')
    
if __name__ == "__main__":
    tokenize()