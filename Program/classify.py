from learning import parse_text, preprocess_text, save_to_txt, count_specific_document
from vocabulary import save_to_csv
from typing import List
from rich import print
from rich.traceback import install
from box import Box
from rich.console import Console
import csv
import pandas as pd # type: ignore
from rich_tools import table_to_df # type: ignore
from rich.table import Table

console = Console()

install(show_locals=True, theme="monokai")

def split_model(title: str) -> Box:
    result = Box({})
    with open(title, 'r', encoding='utf-8') as f:
        next(f)
        next(f)
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0].split(':')[1]
            freq = int(parts[1].split(':')[1])
            log_prob = float(parts[2].split(':')[1])
            result[word] = Box({
                'freq': freq,
                'log_prob': log_prob
            })
    return result

def read_csv(filename: str) -> List[str]:
    rows: List[str] = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows.extend(' '.join(row) for row in reader)
    return rows

def distribute_data() -> None:
    body = open('./F75_train.csv', 'r', encoding='utf-8').read()
    corpus = parse_text(body)

    news_neutral = [x[0] for x in corpus if x[1] == 'neutral']
    news_negative = [x[0] for x in corpus if x[1] == 'positive']
    news_positive = [x[0] for x in corpus if x[1] == 'negative']
    
    print(f'Neutral: {len(news_neutral)}')
    print(f'Negative: {len(news_negative)}')
    print(f'Positive: {len(news_positive)}')
    
    news_dict = Box({
        'neutral': news_neutral, 
        'negative': news_negative,
        'positive': news_positive
    })
    
    pr_news_dict: Box = Box({
        'neutral': [], 
        'negative': [],
        'positive': []
    })

    for key in news_dict:
        for new in news_dict[key]:
            pr_new = preprocess_text(new)
            pr_news_dict[key].append(pr_new)
              
    print(f'Total: {sum(len(news_dict[key]) for key in pr_news_dict)}')
        
    save_to_txt(pr_news_dict.neutral,'./processed/neutral.txt', '\n')
    save_to_txt(pr_news_dict.negative, './processed/negative.txt', '\n')
    save_to_txt(pr_news_dict.positive, './processed/positive.txt', '\n')
    
def classify() -> None:
    body = open('./test.csv', 'r', encoding='utf-8').read()
    corpus = parse_text(body)
    
    preprocessed_text = [preprocess_text(text[0]) for text in corpus]
    lemmatize_words = [list(text.split()) for text in preprocessed_text]
    save_to_csv(lemmatize_words, 'lemmatize_words_2_2.csv')

    size_dict = count_specific_document(corpus)
    size_positive = size_dict[0]
    size_neutral = size_dict[1]
    size_negative = size_dict[2]

    total_len = size_positive + size_neutral + size_negative

    nuetral_model_dic = split_model('./Ficheros/modelo_lenguaje_t_2.txt')
    negative_model_dic = split_model('./Ficheros/modelo_lenguaje_n_2.txt')
    positive_model_dic = split_model('./Ficheros/modelo_lenguaje_p_2.txt')

    correct: int = 0
    incorrect: int = 0

    count_neutral_misses = 0
    count_positive_misses = 0
    count_negative_misses = 0
    CLASIFICATION = 'Clasificación'
    
    table_classification = Table(title=CLASIFICATION)
    table_classification.add_column('Primeros 10 caracteres', justify='center')
    table_classification.add_column('lP en P', justify='center')
    table_classification.add_column('lP en N', justify='center')
    table_classification.add_column('lP en T', justify='center')
    table_classification.add_column(CLASIFICATION, justify='center')
    
    table_resumee = Table(title=CLASIFICATION)
    table_resumee.add_column(CLASIFICATION, justify='center')
    
    for i, doc in enumerate(read_csv('./lemmatize_words_2_2.csv')):
        neutral_prob: float = size_neutral / total_len
        negative_prob: float = size_negative / total_len
        positive_prob: float = size_positive / total_len
        doc_list = doc.split(' ')
        for word in doc_list:
            if word in nuetral_model_dic:
                neutral_prob += nuetral_model_dic[word].log_prob
            if word in negative_model_dic:
                negative_prob += negative_model_dic[word].log_prob
            if word in positive_model_dic:
                positive_prob += positive_model_dic[word].log_prob
                
        positive_prob = abs(positive_prob)
        negative_prob = abs(negative_prob)
        neutral_prob = abs(neutral_prob)

        key = corpus[i][1]
        
        best_val: float = max(neutral_prob, negative_prob, positive_prob)
        best = ''
        
        if best_val == neutral_prob:
            if key == 'neutral':
                correct += 1
                console.print(f'[green]Correct[/green]: {doc}')
            else:
                incorrect += 1
                console.print(f'[red]Incorrect[/red] [blue]neutral-{key}[/blue]: {doc}')
                if key == 'negative':
                    count_negative_misses += 1
                else:
                    count_positive_misses += 1
            best = 'T'
        if best_val == negative_prob:
            if key == 'negative':
                correct += 1
                console.print(f'[green]Correct[/green]: {doc}')
            else:
                incorrect += 1
                console.print(f'[red]Incorrect[/red] [blue]negative-{key}[/blue]: {doc}')
                if key == 'neutral':
                    count_neutral_misses += 1
                else:
                    count_positive_misses += 1
            best = 'N'
        if best_val == positive_prob:
            if key == 'positive':
                correct += 1
                console.print(f'[green]Correct[/green]: {doc}')
            else:
                incorrect += 1
                console.print(f'[red]Incorrect[/red] [blue]positive-{key}[/blue]: {doc}')
                if key == 'neutral':
                    count_neutral_misses += 1
                else:
                    count_negative_misses += 1
            best = 'P'
                    
        table_classification.add_row(doc[:10], str(round(positive_prob, 2)), str(round(negative_prob, 2)), str(round(neutral_prob, 2)), best)
        table_resumee.add_row(best)
        
    print(f'Correct: {correct}, {correct / total_len * 100}%')
    print(f'Incorrect: {incorrect}, {incorrect / total_len * 100}%')
    print(f'Neutral misses: {count_neutral_misses}')
    print(f'Negative misses: {count_negative_misses}')
    print(f'Positive misses: {count_positive_misses}')
    
    df_classification = table_to_df(table_classification)
    df_resumee = table_to_df(table_resumee)
    
    df_classification.to_csv('./Classifier/100_classification.csv', index=False)
    df_resumee.to_csv('./Classifier/100_resumee.csv', index=False)
    
if __name__ == '__main__':
    # distribute_data()
    classify()