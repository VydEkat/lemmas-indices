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
def calculate_similarity(lemma1, gvkey1, word1, lemma2, gvkey2, word2, ic):
      
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
                        'gvkey1': gvkey1,  
                        'word1': word1,  
                        'lemma1': lemma1,  
                        'gvkey2': gvkey2,  
                        'word2': word2,  
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
    # lemmas_df = pd.read_csv('merged_prepostcovid.csv')   
    lemmas_df1 = pd.read_csv('LIST2019_cleaned.csv') 
    lemmas_df2 = pd.read_csv('ADD23.csv')

    lemmas1 = lemmas_df1['lemma19'].tolist()  
    words1 = lemmas_df1['word19'].tolist()  
    gvkeys1 = lemmas_df1['GVKEY'].tolist()  
    lemmas2 = lemmas_df2['lemma23'].tolist()  
    words2 = lemmas_df2['word23'].tolist()  
    gvkeys2 = lemmas_df2['GVKEY'].tolist()  

    lemma_pairs = list(itertools.product(range(len(lemmas1)), range(len(lemmas2))))

    chunk_size = 3000000
    file_index = 0  
    results = []   
    
    # Создаем папку для сохранения результатов
    output_dir = 'similarities_output_list2019_add23'  
    os.makedirs(output_dir, exist_ok=True)  

    total_combinations = len(lemma_pairs)
    with tqdm(total=total_combinations, desc="Обработка пар лемм", unit="пара") as pbar:  
        for i, j in lemma_pairs:  
            lemma1 = lemmas1[i]  
            word1 = words1[i]   
            gvkey1 = gvkeys1[i]  
            lemma2 = lemmas2[j]  
            word2 = words2[j]   
            gvkey2 = gvkeys2[j]  

            result = calculate_similarity(lemma1, gvkey1, word1, lemma2, gvkey2, word2, ic)  
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