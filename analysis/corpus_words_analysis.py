"""
Code used to analyze word count and vocabulary size of the corpus
"""
import pandas as pd
import numpy as np
def get_word_statistics(df, col_name):
  """
  Get the average, median, and standard deviation of the word count for a column in a DataFrame
  """
  column = df[col_name].dropna()
  word_counts = column.apply(lambda x: len(x.split()))

  results = {
      "avg_word_count": np.mean(word_counts),
      "median_word_count": np.median(word_counts),
      "std_dev_word_count": np.std(word_counts)
  }

  return results

def get_unqiue_vocabs(df, col_name):
  """
  Gets the set of unique vocabularies in a column of a DataFrame
  """
  column = df[col_name].dropna()
  vocab_set = set()
  for sentence in column:
    vocab_set.update(sentence.split())

  return vocab_set

def get_vocab_statistics(df, col_name):
  """
  Gets the average, median, and standard deviation of the vocabulary count for a column in a DataFrame
  """
  column = df[col_name].dropna()
  vocab_counts = column.apply(lambda x: len(set(x.split())))

  results = {
      "avg_vocab_count": np.mean(vocab_counts),
      "median_vocab_count": np.median(vocab_counts),
      "std_dev_vocab_count": np.std(vocab_counts),
      "vocab_size": len(get_unqiue_vocabs(df, col_name)),
      "normalized_vocab_size": len(get_unqiue_vocabs(df, col_name))/len(column)
  }

  return results

def print_statistics(df, df_name, target_cols):
    
    """
    Helper function wich calls all previous functions and prints the results
    """
    print(f"Statistics for DataFrame: {df_name}")
    for col_name in target_cols:
        print(f"Column: {col_name}")
        word_stats = get_word_statistics(df, col_name)
        vocab_stats = get_vocab_statistics(df, col_name)
        print("Word Count Statistics:")
        print("  Average:", word_stats["avg_word_count"])
        print("  Median:", word_stats["median_word_count"])
        print("  Standard Deviation:", word_stats["std_dev_word_count"])
        print("Vocabulary Count Statistics:")
        print("  Average:", vocab_stats["avg_vocab_count"])
        print("  Median:", vocab_stats["median_vocab_count"])
        print("  Standard Deviation:", vocab_stats["std_dev_vocab_count"])
        print("  Vocabulary Size:", vocab_stats["vocab_size"])
        print("  Normalized Vocabulary Size:", vocab_stats["normalized_vocab_size"])
        print()


"""
Assuming you have loaded in the DataFrames you wish to anayze, df, val, and test
"""
print_statistics(df, "df", ['Sentence1', 'Sentence2', 'Explanation_1'])
print_statistics(val, "val", ['Sentence1', 'Sentence2', 'Explanation_1', 'Explanation_2', 'Explanation_3'])
print_statistics(test, "test", ['Sentence1', 'Sentence2', 'Explanation_1', 'Explanation_2', 'Explanation_3'])