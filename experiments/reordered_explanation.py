def permute_words(explanation):
    words = explanation.split()
    random.shuffle(words)
    return ' '.join(words)