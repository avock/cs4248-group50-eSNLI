import random, math

def mask_words(sentence):
    MASKING_PERCENTAGE = 0.3
    words = sentence.split()

    num_words_to_mask = min(3, math.floor(MASKING_PERCENTAGE * len(words)))
    words_to_mask_indices = random.sample(range(len(words)), num_words_to_mask)

    for idx in words_to_mask_indices:
        words[idx] = '[MASK]'

    masked_sentence = ' '.join(words)

    return masked_sentence

"""
Assuming you have already loaded DataFrames df, val_df and test_df, and Explanation_1 is the column you wish to mask.
"""
df['Explanation_1'] = df['Explanation_1'].astype(str).apply(mask_words)
val_df['Explanation_1'] = val_df['Explanation_1'].astype(str).apply(mask_words)
test_df['Explanation_1'] = test_df['Explanation_1'].astype(str).apply(mask_words)
