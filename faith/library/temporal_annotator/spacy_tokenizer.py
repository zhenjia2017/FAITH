"""Tokenizer that is backed by spaCy (spacy.io).
Requires spaCy package and the spaCy english model.
"""
import spacy
from faith.library.temporal_annotator.tokenizer import Tokens, Tokenizer
from faith.library.utils import get_config


class SpacyTokenizer(Tokenizer):
    def __init__(self, config):
        self.model = config["spacy_model"]
        self.nlp = spacy.load(self.model)

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')
        tokens = self.nlp(clean_text)

        data = []
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i].idx
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1].idx
            else:
                end_ws = tokens[i].idx + len(tokens[i].text)

            data.append((
                tokens[i].text,
                text[start_ws: end_ws],
                (tokens[i].idx, tokens[i].idx + len(tokens[i].text)),
                tokens[i].pos_,
                tokens[i].tag_,
                tokens[i].lemma_,
                tokens[i].ent_type_,
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, opts={'non_ent': ''})


if __name__ == "__main__":
    import sys

    # RUN: python ehtqa/faithful_er/evidence_retrieval/wikipedia_retriever/step1_wikipedia_year_retriever_decrapt.py config/timequestionsv2/ehtqa.yml
    if len(sys.argv) < 2:
        raise Exception(
            "Usage: python spacy_tokenizer.py  <PATH_TO_CONFIG> [<SOURCES_STRING>]"
        )

    # load config

    config_path = sys.argv[1]
    config = get_config(config_path)
    tokens = SpacyTokenizer(config)
    part1 = "Yin Shun Personal Died ( aged 99 ) Hualien County Republic of China"
    part2 = "Yin Shun date of death"
    print(tokens.tokenize(part1.lower()).lemmas())
    print(tokens.tokenize(part2.lower()).lemmas())
