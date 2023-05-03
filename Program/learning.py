import math
from typing import List, Tuple, Dict
from rich.console import Console
from rich.traceback import install

from vocabulary import read_txt, parse_text, preprocess_text, save_to_txt

console = Console()
install(show_locals=True, theme="monokai")

def count_words(vocabulary: List[str], corpus: List[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    
    for word in vocabulary:
        result[word] = 0
        for document in corpus:
            words_split = document.split()
            word_count = words_split.count(word)
            if word_count > 0:
                result[word] += word_count
            
    return result

def count_specific_document(corpus: List[List[str]]) -> Tuple[int, int, int]:
    neutral : int = 0
    negative : int = 0
    positive : int = 0
    
    for corpus_document in corpus:
        document_type = corpus_document[1]
        if document_type == 'positive':
            positive += 1
        elif document_type == 'neutral':
            neutral += 1
        else:
            negative += 1   

    return (neutral, negative, positive)

def generate_corpues() -> Tuple[int, int, int]:
    body = open('./train.csv', 'r', encoding='utf-8').read()
    corpus = parse_text(body)

    filtered_neutral = list(filter(lambda x: x[1] == 'neutral', corpus))
    filtered_positive = list(filter(lambda x: x[1] == 'positive', corpus))
    filtered_negative = list(filter(lambda x: x[1] == 'negative', corpus))

    filtered_list = [filtered_neutral, filtered_positive, filtered_negative]

    for corpues in filtered_list:
        finished_list: List[str] = [
            preprocess_text(document[0]) for document in corpues
        ]
        save_to_txt(finished_list, f'./data/corpus_{corpues[1][1]}_2.txt', ', ')
        
    return (len(filtered_neutral), len(filtered_positive), len(filtered_negative))
        
def logaritmic_probability(word_abs_fre: int, corpus_size: int, vocabulary_size: int) -> float:
    return math.log((word_abs_fre + 1) / (corpus_size + vocabulary_size + 1))

def generate_model_file(type: str, corpus_size: int, size_news: int, abs_freq_dict: Dict[str, int], log_prob: Dict[str, float]) -> None:
    with open(f'./Ficheros/{type}.txt', 'w') as f:
        head: str = f'Numero de documentos (noticias) del corpus :{size_news}\n'
        head += f'Numero de palabras del corpus :{corpus_size}\n'
        f.write(head)
        abs_freq_dict['UKN'] = 0
        for word, freq in log_prob.items():
            if abs_freq_dict[word] == 0:
                abs_freq_dict['UKN'] += 1
                continue
            line = f'Palabra:{word} Frec:{abs_freq_dict[word]} LogProb:{freq}\n'
            f.write(line)
        f.write(f'Palabra:UKN Frec:{abs_freq_dict["UKN"]} LogProb:{logaritmic_probability(abs_freq_dict["UKN"], corpus_size, len(abs_freq_dict))}\n')
        
def generate_model() -> None:
    body = open('./train.csv', 'r', encoding='utf-8').read()
    corpus = parse_text(body)
    vocabulary: List[str] = read_txt('./Vocabulary/vocabulary.txt', '\n')
    vocabulary.pop(0)
    
    positive_corpus: List[str] = read_txt('./data/corpus_positive_2.txt', ', ')
    negative_corpus: List[str] = read_txt('./data/corpus_negative_2.txt', ', ')
    neutral_corpus: List[str] = read_txt('./data/corpus_neutral_2.txt', ', ')
    
    neutral_count_dict = count_words(vocabulary, neutral_corpus)
    negative_count_dict = count_words(vocabulary, negative_corpus)
    positive_count_dict = count_words(vocabulary, positive_corpus)
    
    neutral_prob: Dict[str, float] = {}
    negative_prob: Dict[str, float] = {}
    positive_prob: Dict[str, float] = {}
    
    for word in vocabulary:
        t_word_prob = logaritmic_probability(neutral_count_dict[word], len(neutral_corpus), len(vocabulary))
        n_word_prob = logaritmic_probability(negative_count_dict[word], len(negative_corpus), len(vocabulary))
        p_word_prob = logaritmic_probability(positive_count_dict[word], len(positive_corpus), len(vocabulary))
        neutral_prob[word] = t_word_prob
        positive_prob[word] = p_word_prob
        negative_prob[word] = n_word_prob
        
    size_dict = count_specific_document(corpus)
    size_neutral = size_dict[0]
    size_negative = size_dict[1]
    size_positive = size_dict[2]
        
    generate_model_file('modelo_lenguaje_t_2', len(neutral_corpus), size_neutral, neutral_count_dict, neutral_prob)    
    generate_model_file('modelo_lenguaje_n_2', len(negative_corpus), size_negative, negative_count_dict, negative_prob)
    generate_model_file('modelo_lenguaje_p_2', len(positive_corpus), size_positive, positive_count_dict, positive_prob)

if __name__ == "__main__":
    generate_corpues()
    generate_model()