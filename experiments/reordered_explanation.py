import random

def permute_words(explanation):
    words = explanation.split()
    random.shuffle(words)
    return ' '.join(words)

"""
Assuming you have already loaded DataFrames df, val_df and test_df, and Explanation_1 is the column you wish to randomly reorder.
"""
df['Explanation_1'] = df['Explanation_1'].astype(str).apply(permute_words)
val_df['Explanation_1'] = val_df['Explanation_1'].astype(str).apply(permute_words)
test_df['Explanation_1'] = test_df['Explanation_1'].astype(str).apply(permute_words)
