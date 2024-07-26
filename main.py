import itertools  
import pandas as pd  
import nltk  
nltk.download('wordnet_ic') 
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from tqdm import tqdm
import os 

# ic_resnik = wordnet_ic.ic('ic-bnc-resnik-add1.dat')
ic = wordnet_ic.ic('ic-bnc-add1.dat')
# ic = wordnet_ic.ic('ic-brown.dat')
# ic_lin = wordnet_ic.ic('ic-semcor.dat')

# Функция для расчета индексов схожести  
def calculate_similarity(lemma1, lemma2, ic):
      
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
                        'lemma1': lemma1,  
                        'lemma2': lemma2,  
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
    lemmas_df = pd.read_csv('LEMLIST_copy.csv')  
    lemmas_df = lemmas_df.dropna(subset=['lemma'])  
    lemmas = lemmas_df['lemma'].tolist()  

    chunk_size = 800000  
    file_index = 0  
    results = []   
    
    # Создаем папку для сохранения результатов
    output_dir = 'similarities_output'  
    os.makedirs(output_dir, exist_ok=True)  

    total_combinations = len(list(itertools.combinations(lemmas, 2)))  
    with tqdm(total=total_combinations, desc="Обработка пар лемм", unit="пара") as pbar:  
        for lemma1, lemma2 in itertools.combinations(lemmas, 2):  
            result = calculate_similarity(lemma1, lemma2, ic)  
            results.extend(result)  

            if len(results) >= chunk_size:  
                df = pd.DataFrame(results)  
                # Сохраняем файл в созданную папку  
                df.to_csv(os.path.join(output_dir, f'similarities_chunk_{file_index}.csv'), index=False)  
                file_index += 1  
                results = []  

            pbar.update(1)  

    if results:  
        df = pd.DataFrame(results)  
        df.to_csv(os.path.join(output_dir, f'similarities_chunk_{file_index}.csv'), index=False)   

if __name__ == "__main__":  
    main()