from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from .prompt import split_with_pattern, remove_repeated_sequences

def preprocess_logits_for_metrics(logits, labels):
        # the actualy prediction is the first element of the tuple
        # Original Trainer may have a memory leak. This is a workaround to avoid storing too many tensors that are not needed
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

def get_completion_with_ans(completion):
    delimiter = "# Answer"
    sub_delimiters = "Answer"
    answer = ''
    if delimiter in completion:
        # with delimiter
        # 1. split into "solution" and "Answer"
        # 2. remove repeatation for only Answer part.
        #  e.g) # Answer 3alalalal # Answer 3 # Answer 3 ...
        # **postprocess for completion is risky
        completion = completion.split(delimiter)
        sol = completion[0] # Solving Part
        ans = completion[1] # Answer Part

        ans = split_with_pattern(r"#|\n", ans)[0]
        answer = remove_repeated_sequences(ans).strip()
        completion = sol + delimiter + ' ' + answer
    else:
        # without delimiter
        # 1. remove repeatation
        # 2. get last line
        # 3. split with '  ' and sub_delimiters(==special case)
        completion = remove_repeated_sequences(completion)
        last_line = split_with_pattern(r"\n", completion)[-1]
        answer = last_line.split('  ')[-1].strip()
        if sub_delimiters in answer:
            answer = last_line.split(sub_delimiters)[-1].strip()
    return (completion, answer)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auroc': auc
    }