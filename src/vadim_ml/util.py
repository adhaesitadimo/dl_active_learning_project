import numpy as np

def try_iterate(seq):
    iterator = iter(seq)
    failure_count = 0
    
    while True:
        try:
            elem = next(iterator)
            yield elem
        except StopIteration:
            print(f'Failed to handle {failure_count} objects')
            break
        except Exception as e:
            print(e)
            failure_count += 1
    
class FiniteGenerator():
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length
        
    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.gen)
    
class NotFoundError(Exception):
    def __init__(self, search_description, query, key, value):
        super().__init__()
        self.description = search_description
        self.query = query
        self.key = key
        self.value = value
        
    def __str__(self):
        r = self.description
        r += '\nNo matches found'
        r += f'\nSearching for {self.query}'
        r += f'\nSearch space: {self.key}'
        r += f'\nValues used as search criteria: {self.value}'
        return r