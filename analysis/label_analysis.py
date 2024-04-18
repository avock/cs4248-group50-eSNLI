import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

'''
Takes in one row of a dataframe, and returns the number of negation words 
in the Explanation_1 colomn.
'''
def count_negations(row):
    negation_list = ['not', 'cannot', 'never']
    total = 0
    if pd.isna(row['Explanation_1']):
        return 0
    for word in row['Explanation_1'].split():
        if word.lower() in negation_list:
            total += 1
    return total

'''
Takes in one row of a dataframe, and returns the number of 
'not necessarily' usage in Explanation_1.
'''
def count_not_necessary(row):
    if pd.isna(row['Explanation_1']):
        return 0
    total = 0
    if 'not necessarily' in row['Explanation_1']:
        return 1
    return total

'''
Takes in one row of a dataframe, and returns the number of unique common
words that exist in all three columns specified, excluding the stop_words.
'''
def count_similar_words(row, c1, c2, c3, stop_words):
    if pd.isna(row[c1]) or pd.isna(row[c2]) or pd.isna(row[c3]):
        return 0
    s1 = set(row[c1].split())
    s2 = set(row[c2].split())
    s3 = set(row[c3].split())
    s1 = set(word.lower() for word in row[c1].split() if word.lower() not in stop_words)
    s2 = set(word.lower() for word in row[c2].split() if word.lower() not in stop_words)
    s3 = set(word.lower() for word in row[c3].split() if word.lower() not in stop_words)
    return len(s1 & s2 & s3)


"Assuming you have df loaded as the train dataset"
df.dropna()

# processing negations
df2 = df[['Explanation_1', 'gold_label']]
df2['negations'] = df2.apply(count_negations, axis=1)
average_negations = df2.groupby('gold_label')[['negations']].mean()
print(average_negations)

# processing not necessarilys
df2['not_necessary'] = df2.apply(count_not_necessary, axis=1)
average_not_necessary = df2.groupby('gold_label')[['not_necessary']].mean()
print(average_not_necessary)

# processing 3 way similarities
df3 = df[['Sentence1', 'Sentence2', 'Explanation_1', 'gold_label']]
stop_words = set(stopwords.words('english'))
df3['count'] = df3.apply(count_similar_words, args=('Sentence1', 'Sentence2', 'Explanation_1', stop_words), axis=1)
average_count = df3.groupby('gold_label')[['count']].mean()
print(average_count)
