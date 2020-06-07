def _binary_to_bio(tags):
    began = False
    for t in tags:
        if t:
            if began:
                yield 'I'
            else:
                began = True
                yield 'B'
        else:
            began = False
            yield 'O'

def binary_to_bio(tags):
    try:
        batch = []
        for t in tags:
            batch.append(binary_to_bio(t))
        return batch
    except TypeError:
        return list(_binary_to_bio(tags))

def _token_span_to_bio(seq, span):
    start, end = span
    ann = ['O' if i < start or i >= end else 'I' for i, s in enumerate(seq)]
    ann[start] = 'B'
    return ann

def token_span_to_bio(seq, spans):
    try:
        return [token_span_to_bio(seq, s) for s in spans]
    except ValueError:
        return _token_span_to_bio(seq, spans)

def to_tree(sequence, parents):
    # Sometimes the input is ['0-root', '2-dep'] instead of [0, 2]
    parents == [int(parent[0]) if isinstance(parent, str) else parent for parent in parents]

    nodes = [(elem, []) for elem in sequence]
    root_node = None

    for idx, parent in enumerate(parents):
        if parent == 0:
            root_node = nodes[idx]

        parent_idx = parent - 1 # 0 means ROOT, 1 means first token
        nodes[parent_idx][1].append(nodes[idx])

    return root_node

def _char_annotations_as_token_annotations(token_spans, annotation_spans):
    for ann in annotation_spans:
        ann_start, ann_end = ann
        start_token, end_token = None, None

        for tok_idx, (tok_start, tok_end) in enumerate(token_spans):
            if ann_start <= tok_end and tok_start < ann_end - 1:
                start_token = tok_idx
            elif start_token and not end_token:
                end_token = tok_idx

        yield start_token, end_token

def char_annotations_as_token_annotations(token_spans, annotation_spans):
    try:
        batch = []
        for spans in annotation_spans:
            batch.append(char_annotations_as_token_annotations(token_spans, spans))
        return batch
    except ValueError:
        return list(_char_annotations_as_token_annotations(token_spans, annotation_spans))

if __name__ == '__main__':
    print(to_tree(['A', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog'],
                  [4, 4, 4, 5, 0, 9, 9, 9, 5]))