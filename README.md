# lemm indices 

Finding indexes for pairs of lemmas synsets

### restrictions
1. If the part of speech of the word is not in ic, then the pair is not considered and the value is assigned null
2. Only pairs with the same parts of speech are considered
3. To speed up the process, only the first pairs of identical parts of speech are taken, since they are the most popular synsets for a given pair of lemmas
4. Total 130 files
5. Each csv file contains no more than 800,000 pairs of lemmas
6. Some indexes are greater than 1 in modulo, perhaps they need normalization