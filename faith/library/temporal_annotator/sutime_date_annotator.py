from sutime import SUTime
import re
from concurrent.futures import ThreadPoolExecutor
from faith.library.date_normalization import DateNormalization
from faith.library.utils import get_logger
# ten years such as 194X: 1940-1949, 180X: 1800-1809
# one hundred years: 19XX
TIMEX_DECADE_PATTERN = re.compile("^[0-9][0-9][0-9]X$")
TIMEX_CENTURY_PATTERN = re.compile("^[0-9][0-9]XX$")
# YEAR-Season such as "2001-SP","2001-SU","2001-FA","2001-WI"
TIMEX_YEAR_SEASON_PATTERN = re.compile("^[0-9][0-9][0-9][0-9]-([SU|FA|WI|SP]{2}?)$")
# timex date in yymmdd format
TIMEX_YMD_PATTERN = re.compile("^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]$")
TIMEX_YMD_PATTERN_BC = re.compile("^[-][0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]$")
# timex date in yy format
TIMEX_YEAR_PATTERN = re.compile("^[0-9][0-9][0-9][0-9]$")
TIMEX_YEAR_PATTERN_BC = re.compile("^[-][0-9][0-9][0-9][0-9]$")
# timex date in yy-mm format
TIMEX_YM_PATTERN = re.compile("^[0-9][0-9][0-9][0-9]-[0-9][0-9]$")
# timex duration in P2014Y format
TIMEX_PYD_PATTERN = re.compile("^P[0-9][0-9][0-9][0-9]Y$")
# # timex date in THIS P1Y INTERSECT YYYY ("the year of") format
TIMEX_INTERSECT_YEAR = re.compile("^THIS P1Y INTERSECT [0-9][0-9][0-9][0-9]$")
# # timex date in THIS P1Y INTERSECT YYYY-MM-DD ("the year of") format
TIMEX_INTERSECT_YMD = re.compile("^THIS P1Y INTERSECT [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]$")
# exclusive text pattern
SIGNAL_IN_TEXT = re.compile("^[\w\W]*(before|after|prior to|in|start|begin|beginning|end)[\w\W]*$")
YEAR_IN_TEXT = re.compile(
        "^(years|yearly|year|the years|the year|month of the year|the same year|the same day|the day|summer|winter|spring|autumn|fall)$")

class SutimeDate:
    # A date annotated by SUTIME with reference date
    def __init__(self, d):
        self.span = (d['start'], d['end'])
        # date mention text
        self.text = d['text']
        # date type
        self.type = d['type']
        # value
        self.value = d['value']
        # reference date
        self.reference = d['reference']

    def json_dict(self):
        # Simple dictionary representation
        return {'text': self.text,
                'type': self.type,
                'span': self.span,
                'value': self.value,
                'reference': self.reference
                }


"""Can be used for annotating dates in questions and evidences."""


class SutimeAnnotator:
    def __init__(self, config, library):
        self.sutime = SUTime(mark_time_ranges=True, include_range=True)
        self.library = library
        self.reference_time = config["reference_time"]
        self.logger = get_logger(__name__, config)

    # sutime annotation basic method
    def sutime_annotation(self, sentence, reference_time = None):
        if not reference_time:
            reference_time = self.reference_time
        result = self.sutime.parse(sentence, reference_time)
        date_annotations = []
        try:
            for tag in result:
                if "value" in tag:
                    date_annotations.append(SutimeDate(
                        {'value': tag['value'], 'text': tag['text'], 'start': tag['start'], 'end': tag['end'],
                         'type': tag['type'], 'reference': reference_time}))
            # date_annotations = [SutimeDate({'value': tag['value'], 'text': tag['text'], 'start': tag['start'], 'end': tag['end'], 'type': tag['type'], 'reference': reference_date}) for tag in
            # result]
            return [w.json_dict() for w in date_annotations]
        except ValueError:
            self.logger.info("Error! That was no valid tag...", sentence)
            self.logger.info("Error! That was no valid tag...", result)

            # sentences include sentence and ites reference time.

    # for example, ['which violent events happened from 1998 to 1999', '2022']
    def sutime_multithreading(self, sentences):
        with ThreadPoolExecutor(max_workers=5) as executor:
            annotation_sentences = [future.result()
                                    for future in [executor.submit(self.sutime_annotation, sentence, reference_date)
                                                   for sentence, reference_date in sentences
                                                   ]]
        return annotation_sentences

    # convert the annotation result into standard format
    def normalization(self, sentence_annotation_result):
        date_norms = []
        for annotation in sentence_annotation_result:
            if not annotation: return []
            if self.sutime_annotation_error(annotation['text']): continue
            if annotation['type'] == 'DATE':
                result = self.timex_date_pattern(annotation['value'], annotation['reference'])
                if result:
                    date_normalization = DateNormalization(
                        {'text': annotation['text'], 'span': annotation['span'], 'timespan': result[1],
                         'method': 'sutime', 'disambiguation': [(result[0], result[1][0])]})
                    date_norms.append(date_normalization)
            elif annotation['type'] == 'DURATION':
                if isinstance(annotation['value'], dict) and 'begin' in annotation['value'] and 'end' in annotation[
                    'value']:
                    result_begin = self.timex_date_pattern(annotation['value']['begin'], annotation['reference'])
                    result_end = self.timex_date_pattern(annotation['value']['end'], annotation['reference'])
                    if result_begin and result_end:
                        date_normalization = DateNormalization(
                            {'text': annotation['text'], 'span': annotation['span'],
                             'timespan': (result_begin[1][0], result_end[1][1]),
                             'method': 'sutime',
                             'disambiguation': [(result_begin[0], result_begin[1][0]),
                                                (result_end[0], result_end[1][0])]})
                        date_norms.append(date_normalization)
        return [w.json_dict() for w in date_norms]

    # check whether there is signal word such as "during" or "year" in the annotation result
    def sutime_annotation_error(self, annotation_text):
        if SIGNAL_IN_TEXT.match(annotation_text):
            return True
        if YEAR_IN_TEXT.match(annotation_text):
            return True
        return False

    # convert sutime annotation result into standard format
    def sutime_annotation_normalization(self, sentence, reference_time = None):
        if not reference_time:
            reference_time = self.reference_time
        sentence_annotation_result = self.sutime_annotation(sentence, reference_time)
        return self.normalization(sentence_annotation_result)

    # multithread convert sutime annotation result into standard format
    def sutime_annotation_normalization_multithreading(self, sentences):
        annotation_sentences = self.sutime_multithreading(sentences)
        """
        normalize annotation with time span and the date is in standard formats YYYY-MM-DDT00:00:00Z
        """
        sentences_date_norms = []
        for annotations in annotation_sentences:
            sentences_date_norms.append(self.normalization(annotations))

        return sentences_date_norms

    def timex_date_pattern(self, date, reference_time = None):
        # convert date value to standard format as timestamp in WikiData
        """
        Generate range for the timestamp.
        Range of YYYY is [YYYY-01-01, YYYY-12-31]
        Range of YYYY-MM [YYYY-MM-01, YYYY-MM-31]
        Range of YYYY-MM-DD is [YYYY-MM-DD, YYYY-MM-DD]
        """
        if not reference_time:
            reference_time = self.reference_time

        PRESENT_REF_YEAR = reference_time.rsplit('-', 2)[0]
        if TIMEX_YMD_PATTERN.match(date):
            return date, self.library.ymd_timespan(date)

        elif TIMEX_YMD_PATTERN_BC.match(date):
            return date, self.library.ymd_timespan(date)

        elif TIMEX_YEAR_PATTERN.match(date):
            return date, self.library.year_timespan(date)

        elif TIMEX_YEAR_PATTERN_BC.match(date):
            return date, self.library.year_timespan(date)

        elif TIMEX_YM_PATTERN.match(date):
            return date, self.library.ym_timespan(date)

        elif TIMEX_PYD_PATTERN.match(date):
            year = date.replace('P', '').replace('Y', '')
            return year, self.library.year_timespan(year)

        elif TIMEX_INTERSECT_YEAR.match(date):
            year = date.replace('THIS P1Y INTERSECT ', '')
            return year, self.library.year_timespan(year)

        elif TIMEX_INTERSECT_YMD.match(date):
            ymd = date.replace('THIS P1Y INTERSECT ', '')
            return ymd, self.library.ymd_timespan(ymd)

        elif 'PRESENT_REF' in date:
            begin_timestamp = f"{PRESENT_REF_YEAR}-01-01T00:00:00Z"
            end_timestamp = f"{PRESENT_REF_YEAR}-12-31T00:00:00Z"
            return reference_time, (begin_timestamp, end_timestamp)

        elif TIMEX_YEAR_SEASON_PATTERN.match(date):
            year = f"{date.split('-')[0]}"
            begin_timestamp = f"{year}-01-01T00:00:00Z"
            end_timestamp = f"{year}-12-31T00:00:00Z"
            return year, (begin_timestamp, end_timestamp)

        elif TIMEX_DECADE_PATTERN.match(date):
            decade_year = f"{date.rstrip('X')}0"
            begin_timestamp = f"{date.rstrip('X')}0-01-01T00:00:00Z"
            end_timestamp = f"{date.rstrip('X')}9-12-31T00:00:00Z"
            return decade_year, (begin_timestamp, end_timestamp)

        elif TIMEX_CENTURY_PATTERN.match(date):
            century_year = f"{date.rstrip('XX')}00"
            begin_timestamp = f"{date.rstrip('XX')}00-01-01T00:00:00Z"
            end_timestamp = f"{date.rstrip('XX')}99-12-31T00:00:00Z"
            return century_year, (begin_timestamp, end_timestamp)


