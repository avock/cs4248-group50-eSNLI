"""# Static masking"""

masking_percentage = 0.3
def mask_words(sentence):
    words = sentence.split()

    import math
    num_words_to_mask = min(3, math.floor(masking_percentage * len(words)))
    words_to_mask_indices = random.sample(range(len(words)), num_words_to_mask)

    for idx in words_to_mask_indices:
        words[idx] = '[MASK]'

    masked_sentence = ' '.join(words)

    return masked_sentence

df['Explanation_1'] = df['Explanation_1'].astype(str).apply(mask_words)
test_df['Explanation_1'] = test_df['Explanation_1'].astype(str).apply(mask_words)
