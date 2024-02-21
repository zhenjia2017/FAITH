import re
from text_to_num import text2num

ORDINAL_WORDS_LAST = ["latest", "newest", "last"]
ORDINAL_PHRASE_LAST = ["most recent", "most recently"]
ORDINAL_WORDS_OLDEST = ["oldest"]
REGEX_ORDINAL_PATTERN = re.compile("^[0-9]+([st|nd|th|rd]{2}?)$")
ORDINAL_NUMS = [
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteen",
    "fifteenth",
    "sixteenth",
    "seventeen",
    "eighteenth",
    "nineteenth",
    "twentieth",
]
MONTH_WORDS_LIST = [
    "january",
    "jan",
    "february",
    "feb",
    "march",
    "mar",
    "april",
    "apr",
    "may",
    "june",
    "jun",
    "july",
    "jul",
    "august",
    "aug",
    "september",
    "sep",
    "october",
    "oct",
    "november",
    "nov",
    "december",
    "dec",
]


class OrdinalNormalization:
    # A normalized ordinal annotated by Regex Expression
    def __init__(self, d):
        # ordinal word mention span
        self.span = (d["start"], d["end"])
        # ordinal word mention text
        self.text = d["text"]
        # normalized ordinal number
        self.ordinal = d["num"]

    def json_dict(self):
        # Simple dictionary representation for ordinal expression
        return {"text": self.text, "span": self.span, "ordinal": self.ordinal}


# check the next word's pos if JJS for ignoring the 2-grams like "first largest" which are not temporal ordinal.
# check the next word if it is a month word
# check the previous word if it is a month word
def check_previous_next_word(idx, word, words, num, span, tags, ordinal_normalization):
    if idx > 0 and idx < len(words) - 1:
        # check the next word
        if (
            tags[idx + 1] != "JJS"
            and words[idx + 1].lower() not in MONTH_WORDS_LIST
            and words[idx - 1].lower() not in MONTH_WORDS_LIST
        ):
            ordinal_normalization.append([word, num, span])

    # if the word is the last word, check the previous word
    elif idx == len(words) - 1:
        if words[idx - 1].lower() not in MONTH_WORDS_LIST:
            ordinal_normalization.append([word, num, span])

    # if the word is the first word, check the next word
    elif idx == 0:
        if tags[idx + 1] != "JJS" and words[idx + 1].lower() not in MONTH_WORDS_LIST:
            ordinal_normalization.append([word, num, span])


def ordinal_annotation(tokenizer):
    ordinal_normalization = []
    entities = tokenizer.entities()
    words = tokenizer.words()
    offsets = tokenizer.offsets()
    tags = tokenizer.tag()
    ordinal_word_part = []
    # string = ' '.join(words)
    # m = re.search(SERIES_ORDINAL_PHRASE_PATTERN, string)
    # try:
    #     print(m.group(0))
    # except:
    #     print('no match')

    for offset in offsets:
        idx = offsets.index(offset)
        word = words[idx]
        word_lower = word.lower()
        span = offset
        ent_type = entities[idx]
        tag = tags[idx]
        # ignore the ordinal word when it is a date
        # but since SpaCy DATE annotation is not good, we do not check it. If we annotate a date as ordinal, we
        # if ent_type == "DATE": continue
        if tag == "CD":
            if idx + 2 < len(words):
                # check if there is an expression like forty-first
                if tags[idx + 1] == "HYPH":
                    if words[idx + 2].lower() in ORDINAL_NUMS:
                        try:
                            word_num = text2num(word_lower, "en")
                            num = word_num + ORDINAL_NUMS.index(words[idx + 2].lower()) + 1
                            ordinal_word_part.append([words[idx + 2], offsets[idx + 2]])
                            ordinal_normalization.append(
                                [
                                    word + words[idx + 1] + words[idx + 2],
                                    num,
                                    (span[0], offsets[idx + 2][1]),
                                ]
                            )
                        except ValueError:
                            print("Oops! The number can not be converted:", word_lower)

        if word_lower in ORDINAL_NUMS and [word, span] not in ordinal_word_part:
            num = ORDINAL_NUMS.index(word_lower) + 1
            check_previous_next_word(idx, word, words, num, span, tags, ordinal_normalization)

        elif REGEX_ORDINAL_PATTERN.match(word_lower) and [word, span] not in ordinal_word_part:
            num = int(re.split(r"[st|nd|th|rd]", word_lower)[0])
            check_previous_next_word(idx, word, words, num, span, tags, ordinal_normalization)

        elif word_lower in ORDINAL_WORDS_LAST:
            ordinal_normalization.append([word, -1, span])
        elif word_lower in ORDINAL_WORDS_OLDEST:
            ordinal_normalization.append([word, 0, span])
        else:
            for phrase in ORDINAL_PHRASE_LAST:
                phrase_len = len(phrase.split(" "))
                if word_lower == phrase.split(" ")[0]:
                    if idx + phrase_len < len(offsets):
                        word_phrase = " ".join(words[idx : idx + phrase_len])
                        span = (offsets[idx][0], offsets[idx + phrase_len - 1][1])
                        if word_phrase.lower() == phrase:
                            ordinal_normalization.append([word_phrase, -1, span])

    ordinal_annotations = [
        OrdinalNormalization(
            {"text": item[0], "num": item[1], "start": item[2][0], "end": item[2][1]}
        )
        for item in ordinal_normalization
    ]

    return [w.json_dict() for w in ordinal_annotations]
