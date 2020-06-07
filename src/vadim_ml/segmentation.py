class TextSegmentation():
    def __init__(self, text, segments = []):
        self.text = str(text)
        self.segments = list(segments)
        self.check_integrity()
        
    def check_integrity(self):
        """Make sure the segments are inside the text, sorted and don't intersect"""

        inside_warning = 'Segments must be inside the text!'
        sorted_warning = 'Segments must be sorted and not intersect!'

        for span, label in self.segments:
            assert span[0] >= 0, inside_warning
            assert span[1] >= 0, inside_warning
            assert span[0] <= len(self.text), inside_warning
            assert span[1] <= len(self.text), inside_warning

        if len(self.segments) > 1:
            for idx in range(len(self.segments) - 1):
                span, label = self.segments[idx]
                next_span, next_label = self.segments[idx + 1]
                assert next_span[0] >= span[1], sorted_warning
    
    def get_segment(self, pos):
        start = 0

        for span, label in self.segments:
            if pos >= span[0]:
                start = span[0]
            else:
                # pos is before this segment
                return (start, span[0]), None

            if pos >= span[1]:
                start = span[1]
            else:
                # pos is in this segment
                return span, label
        
        # pos is after all segments
        return (start, len(self.text)), None 

    def get_label(self, pos):
        for span, label in self.segments:
            if pos < span[1]:
                if pos >= span[0]:
                    return label
                else:
                    return None
            
        return None
    
    def __iter__(self):
        """Iterate over characters labeled with their corresponding segments"""
        
        segment_iter = iter(self.segments)
        impending_span, impending_label = ((0, 0), None)
        cur_label = None
        
        for i, char in enumerate(self.text):
            boundary = False
            if impending_span[1] == i:
                cur_label = None
                impending_span, impending_label = next(segment_iter, ((-1, -1), None))
            if impending_span[0] == i:
                cur_label = impending_label
                
            yield char, cur_label
            
    def stringify(self, open_tag=None, close_tag=None):
        assert open_tag and close_tag
        if isinstance(open_tag, str):
            open_tag = lambda label: open_tag
            close_tag = lambda label: close_tag
        
        output = ""
        segment_iter = iter(self.segments)
        impending_span, impending_label = ((0, 0), None)
        
        for i, char in enumerate(self.text + ' '): # Dirty hack
            if impending_span[1] == i:
                if impending_label is not None:
                    output += close_tag(impending_label)
                impending_span, impending_label = next(segment_iter, ((-1, -1), None))
            if impending_span[0] == i:
                if impending_label is not None:
                    output += open_tag(impending_label)
                
            output += char
                
        return output[:-1] # Dirty hack

    def __repr__(self):
        open_tag = lambda label: '<' + str(label) + '>'
        close_tag = lambda label: '</' + str(label) + '>'
        return self.stringify(open_tag=open_tag, close_tag=close_tag)

    def html(self):
        open_tag = lambda label: f'<b title="{label}">'
        close_tag = lambda label: f'</b>'
        return self.stringify(open_tag=open_tag, close_tag=close_tag)

    def subdocument(self, span):
        new_segments = []
        subdocument_start, subdocument_end = span

        for (segment_start, segment_end), segment_label in self.segments:
            new_segment_start, new_segment_end = (segment_start, segment_end)

            if new_segment_start >= subdocument_end:
                break
            if new_segment_end > subdocument_end:
                new_segment_end = subdocument_end

            new_segment_start -= subdocument_start
            new_segment_end -= subdocument_start

            if new_segment_start < 0:
                new_segment_start = 0
            if new_segment_end <= 0:
                continue

            new_segments.append(((new_segment_start, new_segment_end), segment_label))

        return TextSegmentation(self.text[subdocument_start:subdocument_end], new_segments)

    def __getitem__(self, indexer):
        if isinstance(indexer, slice):
            if indexer.step and indexer.step != 1:
                raise ValueError('TextSegmentation does not support skip slicing')
            return self.subdocument((indexer.start, indexer.stop))
        elif isinstance(indexer, int):
            return self.text[indexer], self.get_label(indexer)
        else:
            raise TypeError(f'TextSegmentation does not support indexers of type {type(indexer)}')

    def merge(text, segments1, segments2):
        segments1_iter = iter(segments1)
        segments2_iter = iter(segments2)
        segments = []
        
        span1, label1 = next(segments1_iter, (None, None))
        span2, label2 = next(segments2_iter, (None, None))
        
        while True:
            if span2 is None or (span1 is not None and span1[0] < span2[0]):
                if span1 is None:
                    # No more segments in the queue. We're done
                    break
                
                segments.append((span1, label1))
                span1, label1 = next(segments1_iter, (None, None))
            else:
                segments.append((span2, label2))
                span2, label2 = next(segments2_iter, (None, None))
        
        # This constructor also calls check_integrity(), so we can be sure that segments don't overlap
        return TextSegmentation(text, segments)

    def concatenate(segmentations):
        all_text = ''
        all_segments = []
        offset = 0

        for seg in segmentations:
            for (begin, end), label in seg.segments:
                all_segments.append((begin + offset, end + offset), label)
            all_text += seg.text
            offset += len(seg.text)

        return TextSegmentation(all_text, all_segments)

    def replace(self, old_substr, new_substr):
        error_text = 'Replacement string of a different length are not supported at the moment'
        assert len(old_substr) == len(new_substr), error_text

        return TextSegmentation(self.text.replace(old_substr, new_substr), self.segments)
