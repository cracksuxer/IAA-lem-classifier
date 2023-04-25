from ast import Dict
from typing import List, Tuple, Dict
from rich import print
from rich.console import Console
from rich.traceback import install

console = Console()
install(show_locals=True, theme="monokai")

def read_txt(filename: str) -> List[str]:
    with open(filename, 'r') as f:
        return list(f.read().split('\n'))
    
def parse_text(text: str) -> List[List[str]]:
    lines = text.strip().split('\n')
    result: List[List[str]] = []
    for line in lines:
        if line == ',':
            return result
        key_value = line.split(',')
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = key_value[1].strip()
        else:
            for i in range(len(key_value) - 2):
                key_value[0] += f',{key_value[i + 1]}'
            value = key_value[-1].strip()
        result.append([key, value])
    return result

def count_words(words: List[str], corpus: List[List[str]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for word in words:
        result[word] = 0
        for document in corpus:
            count = document[0].lower().count(word)
            if count > 0:
                result[word] += count
                
    return result

def count_specific_document(corpus: List[List[str]]) -> Tuple[int, int, int]:
    positive : int = 0
    neutral : int = 0
    negative : int = 0
    
    for corpus_document in corpus:
        document_body = corpus_document[0]
        document_type = corpus_document[1]
        if document_type == 'positive':
            positive += 1
        elif document_type == 'neutral':
            neutral += 1
        else:
            print(document_type)
            negative += 1   
            
    return (positive, neutral, negative)     
            

def main():
    words = read_txt('./Vocabulary/vocabulary.txt')
    words.pop(0)
    
    body = open('./F75_train.csv', 'r', encoding='utf-8').read()
    corpus = parse_text(body)
    # count_dict = count_words(words, corpus)
    
    positive_count, neutral_count, negative_count = count_specific_document(corpus)
    
    print(f'Neutral: {neutral_count}')
    print(f'Positive: {positive_count}')
    print(f'Negative: {negative_count}')

if __name__ == "__main__":
    main()