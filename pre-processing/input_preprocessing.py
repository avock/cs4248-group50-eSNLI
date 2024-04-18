def select_cols(df, col_list):
    '''
    Select columns from a dataframe
    '''
    return df[col_list]

def combine_sentences(df, col_list):

    results_df = df.copy()
    results_df['combined_text'] = '[CLS]' + results_df[col_list].astype(str).agg('[SEP]'.join, axis=1)
    return results_df


lables = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2
}
"""List of Columns you'd wish to preserve"""
target_cols = ['Sentence1', 'Sentence2', 'Explanation_1', 'gold_label'] # Premise, Hypothesis, Explanation

"""
Assuming you have loaded in DataFrames df, val, and test
"""
df = select_cols(df, target_cols)
val = select_cols(val, target_cols)
test_df = select_cols(test, target_cols)

"""Note that we don't take the last column, which is the label"""
df = combine_sentences(df, target_cols[:-1])
val = combine_sentences(val, target_cols[:-1])
test_df = combine_sentences(test_df, target_cols[:-1])

df['labels'] = df['gold_label'].map(lables)
val['labels'] = val['gold_label'].map(lables)
test_df['labels'] = test_df['gold_label'].map(lables)

"""All DataFrames now have a 'combined_text' column, which is a concatenation of the columns in target_cols"""