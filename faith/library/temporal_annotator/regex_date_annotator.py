import re
from concurrent.futures import ThreadPoolExecutor
from faith.library.date_normalization import DateNormalization
from faith.library.utils import get_logger

REGEX_NUM_YEAR_PATTERN = re.compile("^[0-9][0-9][0-9][0-9]$")
REGEX_NUM_YMD_PATTERN = re.compile("^\d{4}([-|/|.]\d{1,2})([-|/|.]\d{1,2})$")
REGEX_NUM_DMY_PATTERN = re.compile("^\d{1,2}([-|/|.]\d{1,2})([-|/|.]\d{4})$")
REGEX_NUM_MDY_PATTERN = re.compile("^\d{1,2}([-|/|.]\d{1,2})([-|/|.]\d{4})$")
REGEX_NUM_YM_PATTERN = re.compile("^\d{4}[-|/|.]\d{1,2}$")
REGEX_NUM_MY_PATTERN = re.compile("^\d{1,2}[-|/|.]\d{4}$")

REGEX_TEXT_DMY_PATTERN = re.compile("[0-9]+ [A-z]* [0-9][0-9][0-9][0-9]")
# mdy dates: https://en.wikipedia.org/wiki/Template:Use_mdy_dates
REGEX_TEXT_MDY_PATTERN = re.compile("[A-z]* [0-9]+, [0-9][0-9][0-9][0-9]")
REGEX_TEXT_YMD_PATTERN = re.compile("[0-9][0-9][0-9][0-9], [A-z]* [0-9]+")
REGEX_TEXT_MY_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|may|june|july|august|september|october|november|december)\s\d{4}\b")

REGEX_TEXT_DATE_PATTERN_TIMESPAN1 = re.compile(
    r"\d{4},\s\w+\s\d{1,2}(?:\u2013\w+\s)?\d{1,2}")  # 2003, March 20\u2013May 22
REGEX_TEXT_DATE_PATTERN_TIMESPAN2 = re.compile(r"\d{4}\u2013\d{4}")  # 2003\u20132005
REGEX_TEXT_DATE_PATTERN_TIMESPAN3 = re.compile(r"\d{4},\s\w+\s\d{1,2}\u2013\d{1,2}")  # 2003, March 20\u201322
REGEX_TEXT_DATE_PATTERN_TIMESPAN4 = re.compile(r"\d{1,2}\s\w+\s\d{4}\s\u2013\s\d{4}")  # 24 May 2001 \u2013 2008
REGEX_TEXT_DATE_PATTERN_TIMESPAN5 = re.compile(
    r"\d{1,2}\s\w+\s\d{4}\s\u2013\s\d{1,2}\s\w+\s\d{4}")  # 29 May 2000 \u2013 13 July 2000
REGEX_TEXT_DATE_PATTERN_TIMESPAN6 = re.compile(
    r"\w+\s\d{1,2},\s\d{4}\s\u2013\s\w+\s\d{1,2},\s\d{4}")  # May 29, 2000 \u2013 July 13, 2000
REGEX_TEXT_DATE_PATTERN_TIMESPAN7 = re.compile(r"\s\d{4}\sto\s\d{4}\s") #1948 to 2005
REGEX_TEXT_DATE_PATTERN_TIMESPAN8 = re.compile(r"\s\d{4}\suntil\s\d{4}\s") #1948 to 2005
"""Can be used for annotating dates and ordinals in questions and evidences."""

class RegexpAnnotator:
    def __init__(self, config, library):
        # load library
        self.library = library
        self.logger = get_logger(__name__, config)

    def remove_punctuation_in_token(self, token):
        punctuations = ['.', ';', '(', ')', '[', ']', ',']
        for punc in punctuations:
            token = token.rstrip(punc)
            token = token.lstrip(punc)
        return token

    def normalize_ymd_date_pattern(self, year, mm, dd):
        try:
            if len(mm) == 1:
                mm = "0" + mm
            if len(dd) == 1:
                dd = "0" + dd
            if int(mm) == 0:
                date = year
                timespan = self.library.year_timespan(year)
                return date, timespan
            elif int(mm) >= 1 and int(mm) <= 12 and int(dd) == 0:
                date = year + '-' + mm
                timespan = self.library.ym_timespan(date)
                return date, timespan
            elif int(mm) >= 1 and int(mm) <= 12 and int(dd) >= 1 and int(dd) <= 31:
                date = year + '-' + mm + '-' + dd
                timespan = self.library.ymd_timespan(date)
                return date, timespan
        except:
            self.logger.info(f"Failure with normalize {year} {mm} {dd}")
            return None

    def extract_dates_in_text_format(self, string):
        date_norms = []
        # dates in dmy format
        dmy_dates = re.findall(REGEX_TEXT_DMY_PATTERN, string)

        for match in dmy_dates:
            patt_start = string.index(match)
            patt_end = patt_start + len(match)
            result = self.library.convert_date_to_timestamp(match, date_format="dmy")
            if result:
                date = result[0]
                timestamp = result[1]
                date_normalization = DateNormalization(
                    {'text': match, 'span': (patt_start, patt_end), 'timespan': (timestamp, timestamp),
                     'method': 'regex',
                     'disambiguation': [(date, timestamp)]})
                date_norms.append(date_normalization)

        # two dates in timespan1 format
        timespan1 = re.findall(REGEX_TEXT_DATE_PATTERN_TIMESPAN1, string)
        for match in timespan1:
            patt_start = string.index(match)
            patt_end = patt_start + len(match)
            result = self.library.convert_date_to_timestamp(match, date_format="timespan1")
            # timestamp1_str, timestamp1, timestamp2_str, timestamp2
            if result:
                date1 = result[0]
                timestamp1 = result[1]
                date2 = result[2]
                timestamp2 = result[3]
                date_normalization = DateNormalization(
                    {'text': match, 'span': (patt_start, patt_end), 'timespan': (timestamp1, timestamp2),
                     'method': 'regex',
                     'disambiguation': [(date1, timestamp1), (date2, timestamp2)]})
                date_norms.append(date_normalization)

        # two dates in timespan2 format
        timespan2 = re.findall(REGEX_TEXT_DATE_PATTERN_TIMESPAN2, string)
        for match in timespan2:
            patt_start = string.index(match)
            patt_end = patt_start + len(match)
            result = self.library.convert_date_to_timestamp(match, date_format="timespan2")
            if result:
                date1 = result[0]
                timestamp1 = result[1]
                date2 = result[2]
                timestamp2 = result[3]
                date_normalization = DateNormalization(
                    {'text': match, 'span': (patt_start, patt_end), 'timespan': (timestamp1, timestamp2),
                     'method': 'regex',
                     'disambiguation': [(date1, timestamp1), (date2, timestamp2)]})
                date_norms.append(date_normalization)

        # two dates in timespan3 format
        timespan3 = re.findall(REGEX_TEXT_DATE_PATTERN_TIMESPAN3, string)
        for match in timespan3:
            patt_start = string.index(match)
            patt_end = patt_start + len(match)
            result = self.library.convert_date_to_timestamp(match, date_format="timespan3")
            if result:
                date1 = result[0]
                timestamp1 = result[1]
                date2 = result[2]
                timestamp2 = result[3]
                date_normalization = DateNormalization(
                    {'text': match, 'span': (patt_start, patt_end), 'timespan': (timestamp1, timestamp2),
                     'method': 'regex',
                     'disambiguation': [(date1, timestamp1), (date2, timestamp2)]})
                date_norms.append(date_normalization)

        # two dates in timespan4 format
        timespan4 = re.findall(REGEX_TEXT_DATE_PATTERN_TIMESPAN4, string)
        for match in timespan4:
            patt_start = string.index(match)
            patt_end = patt_start + len(match)
            result = self.library.convert_date_to_timestamp(match, date_format="timespan4")
            if result:
                date1 = result[0]
                timestamp1 = result[1]
                date2 = result[2]
                timestamp2 = result[3]
                date_normalization = DateNormalization(
                    {'text': match, 'span': (patt_start, patt_end), 'timespan': (timestamp1, timestamp2),
                     'method': 'regex',
                     'disambiguation': [(date1, timestamp1), (date2, timestamp2)]})
                date_norms.append(date_normalization)

        # two dates in timespan5 format
        timespan5 = re.findall(REGEX_TEXT_DATE_PATTERN_TIMESPAN5, string)
        for match in timespan5:
            patt_start = string.index(match)
            patt_end = patt_start + len(match)
            result = self.library.convert_date_to_timestamp(match, date_format="timespan5")
            if result:
                date1 = result[0]
                timestamp1 = result[1]
                date2 = result[2]
                timestamp2 = result[3]
                date_normalization = DateNormalization(
                    {'text': match, 'span': (patt_start, patt_end), 'timespan': (timestamp1, timestamp2),
                     'method': 'regex',
                     'disambiguation': [(date1, timestamp1), (date2, timestamp2)]})
                date_norms.append(date_normalization)

        # two dates in timespan6 format
        timespan6 = re.findall(REGEX_TEXT_DATE_PATTERN_TIMESPAN6, string)
        for match in timespan6:
            patt_start = string.index(match)
            patt_end = patt_start + len(match)
            result = self.library.convert_date_to_timestamp(match, date_format="timespan6")
            if result:
                date1 = result[0]
                timestamp1 = result[1]
                date2 = result[2]
                timestamp2 = result[3]
                date_normalization = DateNormalization(
                    {'text': match, 'span': (patt_start, patt_end), 'timespan': (timestamp1, timestamp2),
                     'method': 'regex',
                     'disambiguation': [(date1, timestamp1), (date2, timestamp2)]})
                date_norms.append(date_normalization)

        # two dates in timespan6 format
        timespan8 = re.findall(REGEX_TEXT_DATE_PATTERN_TIMESPAN8, string)
        for match in timespan8:
            patt_start = string.index(match)
            patt_end = patt_start + len(match)
            result = self.library.convert_date_to_timestamp(match, date_format="timespan8")
            if result:
                date1 = result[0]
                timestamp1 = result[1]
                date2 = result[2]
                timestamp2 = result[3]
                date_normalization = DateNormalization(
                    {'text': match, 'span': (patt_start, patt_end),
                     'timespan': (timestamp1, timestamp2.replace("-01-01", "-12-31")),
                     'method': 'regex',
                     'disambiguation': [(date1, timestamp1), (date2, timestamp2)]})
                date_norms.append(date_normalization)

        # two dates in timespan6 format
        timespan7 = re.findall(REGEX_TEXT_DATE_PATTERN_TIMESPAN7, string)
        for match in timespan7:
            patt_start = string.index(match)
            patt_end = patt_start + len(match)
            result = self.library.convert_date_to_timestamp(match, date_format="timespan7")
            if result:
                date1 = result[0]
                timestamp1 = result[1]
                date2 = result[2]
                timestamp2 = result[3]
                date_normalization = DateNormalization(
                            {'text': match, 'span': (patt_start, patt_end), 'timespan': (timestamp1, timestamp2.replace("-01-01","-12-31")),
                             'method': 'regex',
                             'disambiguation': [(date1, timestamp1), (date2, timestamp2)]})
                date_norms.append(date_normalization)

        # month year format
        monthyear = re.findall(REGEX_TEXT_MY_PATTERN, string)
        for match in monthyear:
            patt_start = string.index(match)
            patt_end = patt_start + len(match)
            result = self.library.convert_date_to_timestamp(match, date_format="my")
            if result:
                date1 = result[0]
                timestamp1 = result[1]
                date2 = result[2]
                timestamp2 = result[3]
                date_normalization = DateNormalization(
                    {'text': match, 'span': (patt_start, patt_end), 'timespan': (timestamp1, timestamp2),
                     'method': 'regex',
                     'disambiguation': [(date1, timestamp1)]})
                date_norms.append(date_normalization)

        # dates in ymd format
        ymd_dates = re.findall(REGEX_TEXT_YMD_PATTERN, string)
        for match in ymd_dates:
            patt_start = string.index(match)
            patt_end = patt_start + len(match)
            result = self.library.convert_date_to_timestamp(match, date_format="ymd")
            if result:
                date = result[0]
                timestamp = result[1]
                date_normalization = DateNormalization(
                    {'text': match, 'span': (patt_start, patt_end), 'timespan': (timestamp, timestamp),
                     'method': 'regex',
                     'disambiguation': [(date, timestamp)]})
                date_norms.append(date_normalization)

        # dates in mdy format
        mdy_dates = re.findall(REGEX_TEXT_MDY_PATTERN, string)
        for match in mdy_dates:
            patt_start = string.index(match)
            patt_end = patt_start + len(match)
            result = self.library.convert_date_to_timestamp(match, date_format="mdy")
            if result:
                date = result[0]
                timestamp = result[1]
                date_normalization = DateNormalization(
                    {'text': match, 'span': (patt_start, patt_end), 'timespan': (timestamp, timestamp),
                     'method': 'regex',
                     'disambiguation': [(date, timestamp)]})
                date_norms.append(date_normalization)

        return [w.json_dict() for w in date_norms]

    def extract_date_in_num_format(self, string):
        date_norms = []
        tokens = string.split(" ")
        for token in tokens:
            token_withno_punc = self.remove_punctuation_in_token(token)
            token_start = string.index(token_withno_punc)
            token_end = token_start + len(token_withno_punc)

            if REGEX_NUM_YEAR_PATTERN.match(token_withno_punc):
                timestamp = self.library.convert_year_to_timestamp(token_withno_punc)
                date_normalization = DateNormalization(
                    {'text': token_withno_punc, 'span': (token_start, token_end),
                     'timespan': (timestamp, f"{token_withno_punc}-12-31T00:00:00Z"),
                     'method': 'regex',
                     'disambiguation': [(token_withno_punc, timestamp)]})
                date_norms.append(date_normalization)

            if REGEX_NUM_YMD_PATTERN.match(token_withno_punc):
                year = re.split(r'[-|.|/]', token_withno_punc)[0]
                mm = re.split(r'[-|.|/]', token_withno_punc)[1]
                dd = re.split(r'[-|.|/]', token_withno_punc)[2]
                result = self.normalize_ymd_date_pattern(year, mm, dd)
                if result:
                    date = result[0]
                    timespan = result[1]
                    timestamp = timespan[0]
                    date_normalization = DateNormalization(
                        {'text': token_withno_punc, 'span': (token_start, token_end), 'timespan': timespan,
                         'method': 'regex',
                         'disambiguation': [(date, timestamp)]})
                    date_norms.append(date_normalization)

            elif REGEX_NUM_MDY_PATTERN.match(token_withno_punc):
                year = re.split(r'[-|.|/]', token_withno_punc)[2]
                mm = re.split(r'[-|.|/]', token_withno_punc)[0]
                dd = re.split(r'[-|.|/]', token_withno_punc)[1]
                result = self.normalize_ymd_date_pattern(year, mm, dd)
                if result:
                    date = result[0]
                    timespan = result[1]
                    timestamp = timespan[0]
                    date_normalization = DateNormalization(
                        {'text': token_withno_punc, 'span': (token_start, token_end), 'timespan': timespan,
                         'method': 'regex',
                         'disambiguation': [(date, timestamp)]})
                    date_norms.append(date_normalization)

                else:
                    year = re.split(r'[-|.|/]', token)[2]
                    mm = re.split(r'[-|.|/]', token)[1]
                    dd = re.split(r'[-|.|/]', token)[0]
                    result = self.normalize_ymd_date_pattern(year, mm, dd)
                    if result:
                        date = result[0]
                        timespan = result[1]
                        timestamp = timespan[0]
                        date_normalization = DateNormalization(
                            {'text': token_withno_punc, 'span': (token_start, token_end), 'timespan': timespan,
                             'method': 'regex',
                             'disambiguation': [(date, timestamp)]})
                        date_norms.append(date_normalization)

        return [w.json_dict() for w in date_norms]

    # multithread annotate sentences using regular expression and normalize them into standard format
    def regex_annotation_normalization_multithreading(self, string_refers):
        with ThreadPoolExecutor(max_workers=5) as executor:
            annotation_sentences = [future.result()
                                    for future in [executor.submit(self.regex_annotation_normalization, string)
                                                   for string, reference_time in string_refers
                                                   ]]
        return annotation_sentences

    # annotate sentences using regular expression and normalize them into standard format
    def regex_annotation_normalization(self, string):
        """
        Extract dates in text (added to entities).
        First, text is searched for text, then the dates
        are brought into a compatible format (timestamps).
        """

        date_norms_text = self.extract_dates_in_text_format(string)
        date_norms_num = self.extract_date_in_num_format(string)
        date_norms = self.remove_duplicate_matched(date_norms_text, date_norms_num)
        return date_norms

    # remove duplicate annotation results
    def remove_duplicate_matched(self, annotations_text, annotations_number):
        disambiguations = {}
        for item in annotations_text:
            span = item['span']
            disambiguations[span] = item
        for item in annotations_number:
            span = item['span']
            disambiguations[span] = item

        start_end = list(disambiguations.keys())
        if len(start_end) > 1:
            for i in range(0, len(start_end) - 1):
                for j in range(i + 1, len(start_end)):
                    if not self.check_overlap(start_end[i], start_end[j]):
                        continue
                    else:
                        lengthi = start_end[i][1] - start_end[i][0]
                        lengthj = start_end[j][1] - start_end[j][0]
                        if lengthj >= lengthi:
                            if start_end[i] in disambiguations:
                                disambiguations.pop(start_end[i])
                        else:
                            if start_end[j] in disambiguations:
                                disambiguations.pop(start_end[j])

        return list(disambiguations.values())

    # check whether two results are overlap
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


if __name__ == "__main__":

    import argparse

    "Usage: python faith/library/temporal_annotator/regex_date_annotator.py -f <FUNCTION> -e <REFERENCE_TIME> -t <TEXT>"
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--function', type=str, default='sutime')
    argparser.add_argument('-t', '--text', type=str, default='what was the current population of japan?')
    argparser.add_argument('-e', '--reference', type=str, default='2023-01-01')
    args = argparser.parse_args()
    function = args.function
    reference_time = args.reference
    text = args.text
    if function == 'regex':
        regex = RegexpAnnotator()
        result = regex.regex_annotation_normalization(text)
        print(result)
