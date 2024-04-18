from transformers_interpret import SequenceClassificationExplainer
import time
import csv

VERBOSE = False
PROGRESS_VERBOSE = True

def normalize_dict(dictionary):

    """
    Normalize a dictionary to sum to 1.
    """
    total_abs_sum = sum(abs(value) for value in dictionary.values())
    normalized_dict = {key: value / total_abs_sum for key, value in dictionary.items()}

    for key, value in dictionary.items():
        if value < 0:
            normalized_dict[key] *= -1

    return normalized_dict

"""
Computes how much each segment contributes to the output of the model.
Each segment is seperated by a [SEP] token.

Input:
  - df: dataframe to test on, ensrue it contains the cols "combined_text" and "labels"
  - model: target fine-tuned model
  - col_names: list of column names to compute attributions for
  - tokenizer
  - (Optional) flag to normalize the attributions to sum to 1
Output: A dict containing a break down of the weightage each segment contributes to the output of the model.
"""
def compute_segment_attributions(df, model, tokenizer, col_names, normalize=False):
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    results, false_results = {}, {}

    start_time = time.time()
    for index, row in df.iterrows():
        text = row['combined_text']
        true_label = row['labels']
        if index % 100 == 0:
            print(f"Processing index {index}, Time: {time.time() - start_time}") if PROGRESS_VERBOSE else None
            start_time = time.time()

        word_attributions = cls_explainer(text)
        tokens = tokenizer.tokenize(text)


        segments = text.split('[SEP]')
        segments[0] = segments[0].replace('[CLS]', '').strip()
        print(segments) if VERBOSE else None

        attribution_sums = {}
        token_index = 1

        for idx, segment in enumerate(segments):

            segment_tokens = tokenizer.tokenize(segment.strip())
            end_index = token_index + len(segment_tokens) + 1

            print(f"Attributes: {word_attributions[token_index:end_index]}") if VERBOSE else None # Remember that the tokenizer adds [CLS] and [SEP] as well

            segment_attributions = []


            for word, attr in word_attributions[token_index:end_index]:
              if word not in ['[CLS]', '[SEP]']:
                print(f"Using the word {word}") if VERBOSE else None
                segment_attributions.append(attr)

            sum_attributions = sum(segment_attributions)
            attribution_sums[col_names[idx]] = sum_attributions
            token_index = end_index

        if cls_explainer.predicted_class_index == true_label:
            results[index] = normalize_dict(attribution_sums) if normalize else attribution_sums
        else:
            results[index] = {}
            false_results[index] = normalize_dict(attribution_sums) if normalize else attribution_sums

    return results, false_results

"""
Assuming you have loaded in the target model and it's tokenizer, along with col_names which is the list of segments you want to compute attributions for.

Returns two dictionaries, results and false_results; 
Results contains the attributions for ALL samples, while false_results only contains the attributions for the incorrectly classified samples, used for analysis.
"""
results, false_results = compute_segment_attributions(test_df, model, tokenizer, col_names, normalize=True)

"""Methods of saving results into a CSV file is demonstrated below"""

file_path = f"/PATH/TO/SAVE/FILE/segment_sequence_results.csv"
fieldnames = ['Index', 'Premise', 'Hypothesis', 'Explanation'] # note the extra 'Index' which was not present in col_names

with open(file_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for index, row_data in results.items():
        writer.writerow({'Index': index, **row_data})
