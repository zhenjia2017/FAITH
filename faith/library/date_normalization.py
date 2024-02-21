class DateNormalization:
    # A normalized date annotated by SuTime or Regex Expression
    def __init__(self, d):
        # normalized time span
        self.timespan = d['timespan']
        # date mention text
        self.text = d['text']
        # date mention text span
        self.span = d['span']
        # date annotation method
        self.method = d['method']
        # date timestamp
        self.disambiguation = d['disambiguation']

    def json_dict(self):
        # Simple dictionary representation
        return {
            'text': self.text,
            'method': self.method,
            'timespan': self.timespan,
            'span': self.span,
            'disambiguation': self.disambiguation
        }