import numpy as np

def collate(seq):
    elems = list(seq)
    try:
        return np.stack(elems)
    except ValueError:
        return list(elems)

def batch_n(seq, collate_fn = collate, batch_size=500):
    """
    Split a sequence into batches of size batch_size. I know.
    """

    batch_size = int(batch_size)

    departure_queue = []
    for elem in seq:
        departure_queue.append(elem)
        if len(departure_queue) == batch_size:
            yield collate_fn(departure_queue)
            departure_queue = []
            
    if departure_queue:
        yield collate_fn(departure_queue)

def batch_by(seq, key, collate_fn = collate, megabatch_size=10000, batch_size=500):
    """
    Split a sequence into batches with equal value of key function
    No batch will be larger than batch_size
    """

    megabatch_size = int(megabatch_size)
    batch_size = int(batch_size)

    batches = {}
    megabatch_count = 0
    
    for elem in seq:
        param = key(elem)
        
        if param not in batches:
            batches[param] = []
        batches[param].append(elem)
        
        megabatch_count += 1
        if megabatch_count == megabatch_size:
            for param in batches:
                for batch in batch_n(batches[param], collate_fn=collate_fn, batch_size=batch_size):
                    yield batch
            batches = {}
            megabatch_count = 0
                
    for param in batches:
        for batch in batch_n(batches[param], collate_fn=collate_fn, batch_size=batch_size):
            yield batch