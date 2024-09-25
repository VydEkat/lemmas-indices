import itertools  
import pandas as pd  
import nltk  
# nltk.download('wordnet_ic') 
# nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from tqdm import tqdm
import os 

# ic_resnik = wordnet_ic.ic('ic-bnc-resnik-add1.dat')
# ic = wordnet_ic.ic('ic-brown.dat')
# ic_lin = wordnet_ic.ic('ic-semcor.dat')
ic = wordnet_ic.ic('ic-bnc-add1.dat')

# Функция для расчета индексов схожести  
def calculate_similarity(lemma1, lemma2, serial_no, word, ic):
      
    # Получаем синсеты для лемм  
    synsets1 = wn.synsets(lemma1)  
    synsets2 = wn.synsets(lemma2)  

    first_result = {}  

    for syn1 in synsets1:  
        for syn2 in synsets2:  
            # Сравниваем только одинаковые части речи  
            if syn1.pos() == syn2.pos():   
                if (lemma1, lemma2) not in first_result:  
                    similarity_data = {
                        'serial_no': serial_no,
                        'word' : word,   
                        'lemma-snword': lemma1,  
                        'lemma-gsword': lemma2,
                        'path_similarity': wn.path_similarity(syn1, syn2) if syn1 and syn2 else None,  
                        'leacock_chodorow': wn.lch_similarity(syn1, syn2) if syn1 and syn2 else None,  
                        'wu_palmer': wn.wup_similarity(syn1, syn2) if syn1 and syn2 else None,  
                        'resnik': wn.res_similarity(syn1, syn2, ic) if syn1 and syn2 and syn1.pos() in ic else None,  
                        'jiang_conrath': wn.jcn_similarity(syn1, syn2, ic) if syn1 and syn2 and syn1.pos() in ic else None,  
                        'lin': wn.lin_similarity(syn1, syn2, ic) if syn1 and syn2 and syn1.pos() in ic else None  
                    }  
                    first_result[(lemma1, lemma2)] = similarity_data  
                    break  # Прерываемся, чтобы взять только первый результат  

        if (lemma1, lemma2) in first_result:  
            break

    return list(first_result.values())  

# Основная функция  
def main():  

    lemmas_df = pd.read_csv('snword-gsword-cleaned.csv')

    results = []

    total_combinations = len(lemmas_df)  # Полное количество строк в DataFrame  
    with tqdm(total=total_combinations, desc="Обработка пар лемм", unit="пара") as pbar:  
        for index, row in lemmas_df.iterrows():  
            lemma1 = row['lemma-snword']  
            lemma2 = row['lemma-gsword']  
            serial_no = row['serial_no']   
            word = row['word']
            
            result = calculate_similarity(lemma1, lemma2, serial_no, word, ic)   
            results.extend(result)  
            pbar.update(1)  

    # Сохранение всех результатов в один файл  
    output_dir = 'similarities_output_snword-gsword'  
    os.makedirs(output_dir, exist_ok=True)  
    results_df = pd.DataFrame(results)  
    results_df.to_csv(os.path.join(output_dir, 'similarities_output_snword-gsword.csv'), index=False)  

if __name__ == "__main__":  
    main()  