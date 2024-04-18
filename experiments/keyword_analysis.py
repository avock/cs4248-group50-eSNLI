import pandas as pd
import time, datetime, numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
import torch
from torch.utils.data import Dataset, DataLoader
import time


"""
Computes the attribution score of highlighted keywords in the test case.

Input:
  - input_attribution: List of (token, attribution) tuples for the test input.
  - annotated_attributions: List of (token, attribution) tuples for the annotated premise and annotated hypothesis.

Output: List of (token, attribution) tuples for the highlighted keywords/annotated keywords in the premise and hypothesis.
"""

def extract_marked_token_attributions(input_attribution, annotated_attributions):
    index_1 = 0
    index_2 = 0
    desired_version = []
    for i in range(len(annotated_attributions)):
      if index_1 < len(input_attribution):
        if input_attribution[index_1][0] != '*':
          index_1 += 1
          index_2 += 1
        else:
          index_1 += 1
          while input_attribution[index_1][0] != '*':
            desired_version.append(annotated_attributions[index_2])
            index_1 += 1
            index_2 += 1
          index_1 += 1

    return desired_version

"""
Computes the highlighted keyword score for each test case. 

Input:
  - df: Preprocessed test dataset with columns ['combined_text', 'labels', 'annotated_premise', 'annotated_hypothesis']
  - model: Initialised BERT model with saved weights
  - tokenizer: BERT tokenizer

Output: List of highlighted keyword score for each test case. 
"""

def compute_highlighted_keyword_scores(df, model, tokenizer):
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    results = []

    for index, row in df.iterrows():
        text = row['combined_text']
        true_label = row['labels']
        highlight_premise = row['annotated_premise']
        highlight_hypothesis = row['annotated_hypothesis']

        text = text.replace("[CLS]", "").replace("[SEP]", "")
        print(f"\nProcessing index {index} with text: {text}")
        full_attributions = cls_explainer(text)
        
        if cls_explainer.predicted_class_index == true_label:
          annotated_attributions = cls_explainer(highlight_premise + " " + highlight_hypothesis) # Premise before hypothesis (change order if input order is changed)

          desired_attributions = extract_marked_token_attributions(annotated_attributions, full_attributions)

          desired_attribution_score = 0
          total_positive_attribution_score = 0

          for pair in desired_attributions:
            print(f"Annotated Word: {pair[0]}\nAttribution: {pair[1]}")
            print("====================================================")
            desired_attribution_score += pair[1]

          for pair in full_attributions:
            if pair[1] > 0:
              total_positive_attribution_score += pair[1]

          print(f"Total Desired Attribution: {desired_attribution_score}")
          print(f"Total Attribution Score: {total_positive_attribution_score}")
          print(f"Test Case Score: {desired_attribution_score/total_positive_attribution_score}")

          results.append(desired_attribution_score/total_positive_attribution_score)

    return results

# Format is [CLS] Premise [SEP] Hypothesis [SEP] anything else
# VERY IMPORTANT: IT MUST BE PREMISE BEFORE HYPOTHESIS (if not then change the order where indicated)
results = compute_segment_attributions(df, model, tokenizer)
print("\nFinal Attribution results:", results)


