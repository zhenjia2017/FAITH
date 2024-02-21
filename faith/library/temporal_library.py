import json
from faith.library.temporal_annotator.regex_date_annotator import RegexpAnnotator
from faith.library.temporal_annotator.ordinal_annotator import ordinal_annotation
from faith.library.temporal_annotator.spacy_tokenizer import SpacyTokenizer
from faith.library.utils import get_logger
from faith.library.temporal_annotator.sutime_date_annotator_client import SutimeClient

class TemporalValueAnnotator:
    def __init__(self, config, library):
        self.logger = get_logger(__name__, config)
        self.library = library
        # You could replace the server hostname and port according to your configuration
        self.sutime = SutimeClient(host="http://localhost", port="7779")
        self.regex = RegexpAnnotator(config, library)
        self.reference_time = config["reference_time"]
        self.tokenizer = SpacyTokenizer(config)

    def date_ordinal_annotator(self, string, reference_time = None, date_tag_method = "regex"):
        if not reference_time:
            reference_time = self.reference_time
        date_annotator_result = self.date_annotator(string, reference_time, date_tag_method)
        ordinal_annotator_result = self.ordinal_annotator(string, date_annotator_result)
        return (date_annotator_result, ordinal_annotator_result)

    def date_annotator_multithread(self, string_refers, tag_method = "regex"):
        annotation_dates = []
        if tag_method == 'sutime_regex':
            annotate_strings = self.sutime.sutime_annotation_normalization_multithreading(string_refers)
            print(annotate_strings)
            regex_results = self.regex.regex_annotation_normalization_multithreading(string_refers)
            for i in range(len(annotate_strings)):
                annotation_dates.append(self.remove_duplicate_matched_dates(annotate_strings[i], regex_results[i]))
        elif tag_method == 'sutime':
            annotation_dates = self.sutime.sutime_annotation_normalization_multithreading(string_refers)
            print(annotation_dates)
        else:
            annotation_dates = self.regex.regex_annotation_normalization_multithreading(string_refers)
        return annotation_dates

    def date_annotator(self, string, reference_time = None, tag_method = "regex"):
        """
        Can be used for annotating dates in questions and evidences.
        :param string: a question or a sentence from evidences
        :param reference_time: question creation date or evidence creation date
        :return: data annotation result, a list of dictionary including
        {
                'text': date text,
                'method': annotation method,
                'timespan': normalized timespan,
                'span': text span,
                'disambiguation': timestamp of date
                }
        """
        # different data annotation method chosen in config
        # 0: sutime and regular expression
        # 1: regular expression
        # 2: sutime
        if not reference_time:
            reference_time = self.reference_time
        if tag_method == 'sutime_regex':
            #both sutime and regular expression
            regex_result = self.regex.regex_annotation_normalization(string)
            sutime_result = self.sutime.sutime_annotation_normalization(string, reference_time)
            print(sutime_result)
            return self.remove_duplicate_matched_dates(sutime_result, regex_result)
        elif tag_method == 'regex':
            #regular expression
            return self.regex.regex_annotation_normalization(string)
        elif tag_method == 'sutime':
            #sutime
            sutime_result = self.sutime.sutime_annotation_normalization(string, reference_time)
            print(sutime_result)
            return sutime_result

    def ordinal_annotator(self, string, date_annotator_result):
        """
        Can be used for annotating ordinals in questions and evidences.
        annotate ordinal.
        :param string: a question or a sentence from evidences
        :param date_annotator_result:
        :return: ordinal annotation result: a list of dictionary including
                {'text': ordinal text,
                'span': text span,
                'ordinal': ordinal normalization number
                }
        """
        tokenized_string = self.tokenizer.tokenize(string)
        ordinal_result = ordinal_annotation(tokenized_string)
        ordinal_annotator_result = self.remove_ordinal_in_date(ordinal_result, date_annotator_result)
        return ordinal_annotator_result

    def remove_duplicate_matched_dates(self, sutime_dates, regex_dates):
        """
        # remove regex date tags in sutime date
        :param sutime_dates:
        :param regex_dates:
        :return:
        """
        sutime_spans = {tuple(item['span']): item for item in sutime_dates}
        regex_spans = {tuple(item['span']): item for item in regex_dates}
        sutime_spans.update(regex_spans)
        start_end = list(sutime_spans.keys())
        if len(start_end) > 1:
            for i in range(0, len(start_end) - 1):
                for j in range(i + 1, len(start_end)):
                    if not self.check_overlap(start_end[i], start_end[j]):
                        continue
                    else:
                        lengthi = start_end[i][1] - start_end[i][0]
                        lengthj = start_end[j][1] - start_end[j][0]
                        if lengthj >= lengthi:
                            if start_end[i] in sutime_spans:
                                sutime_spans.pop(start_end[i])
                        else:
                            if start_end[j] in sutime_spans:
                                sutime_spans.pop(start_end[j])
        return list(sutime_spans.values())

    def remove_ordinal_in_date(self, ordinals, dates):
        """
        remove ordinal tags in date
        :param ordinals:
        :param dates:
        :return:
        """
        date_span = []
        priority_spans = [item['span'] for item in dates]
        spans = [item['span'] for item in ordinals]
        if spans and priority_spans:
            for priority_item in priority_spans:
                for item in spans:
                    if item[0] >= priority_item[0] and item[1] <= priority_item[1]:
                        date_span.append(item)

        return [item for item in ordinals if item['span'] not in date_span]

    # check if there are different mentions which are overlapped and matched to one date
    def check_overlap(self, rangei, rangej):
        start1 = rangei[0]
        end1 = rangei[1]
        start2 = rangej[0]
        end2 = rangej[1]

        if end1 < start2 or start1 > end2:
            return False
        elif start1 == start2 or end1 == end2 or start1 == end2 or start2 == end1:
            return True
        elif start1 < start2 and end1 > start2:
            return True
        elif start1 > start2 and start1 < end2:
            return True