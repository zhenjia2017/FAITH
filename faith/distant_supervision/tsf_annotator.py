import time
from faith.library.utils import get_logger
from faith.library.string_library import StringLibrary
from faith.faithful_er.evidence_retrieval.clocq_er import ClocqRetriever
from faith.library.temporal_library import TemporalValueAnnotator
from faith import evaluation as evaluation


class TSFAnnotator:
    def __init__(self, clocq, config):
        self.clocq = clocq
        self.config = config
        self.logger = get_logger(__name__, config)
        self.string_lib = StringLibrary(config)
        self.temporal_value_annotator = TemporalValueAnnotator(config, self.string_lib)
        self.retriever = ClocqRetriever(config, self.temporal_value_annotator)
        self.type_relevance_cache = dict()
        self.sources = config["ds_sources"]

    def process_instance(self, instance):
        """
        Get relevant entities for the question.
        """
        question = instance["Question"]
        answers = self.string_lib.format_answers(instance)
        instance["silver_tsf"] = None
        instance["question"] = question
        instance["answers"] = answers
        evidences, question_entities = self.retriever.retrieve_evidences(question, self.sources)
        answering_evidences = self._get_answering_evidences(evidences, answers)
        if not answering_evidences:
            return False
        disambiguation_tuples = self._get_answer_connecting_disambiguations(
            question_entities, answering_evidences, answers
        )
        tsf = self._construct_tsf(instance, disambiguation_tuples)
        instance["silver_tsf"] = tsf

        return True

    def _construct_tsf(self, instance, disambiguation_tuples):
        """
        Extract a structured representation for the question.
        """
        question = instance["Question"]
        answers = instance["answers"]
        #only consider one signal
        signal = instance["Temporal signal"][0]
        #only consider one type
        question_type = instance["Temporal question type"][0]
        # derive TSF answer type
        # only consider one answer type
        tsf_answer_type = self._get_answer_type(question, answers)

        # remove entity surface forms from question
        question = self._remove_surface_forms(question, disambiguation_tuples)

        # derive TSF entities
        tsf_entities = list()
        for _, surface_forms, _ in disambiguation_tuples:
            for surface_form in surface_forms:
                if surface_form not in tsf_entities:
                    tsf_entities.append(surface_form)

        # derive TSF relation
        if self.config["ds_remove_stopwords"]:
            words = self.string_lib.get_question_words(question, ner=None)
            tsf_relation = " ".join(words)
            tsf_relation = self._normalize_relation_tsf(tsf_relation)
        else:
            # remove symbols
            tsf_relation = self._normalize_relation_tsf(question)

        # create TSF
        # keep tsf as dictionary
        tsf = {
            "entity": tsf_entities,
            "relation": tsf_relation,
            "answer_type": tsf_answer_type,
            "temporal_signal": signal,
            "categorization": question_type
        }
        return tsf

    def _get_answering_evidences(self, evidences, answers):
        """
        Among the given evidences, extract the subset that has the answers. If the
        disambiguated item is already an answer, no need to return full 1-hop of
        such an answer (full 1-hop has answer => is answering).
        """
        # check whether answer is in facts
        answer_ids = [answer["id"] for answer in answers]
        _, answering_evidences = evaluation.answer_presence(evidences, answers, relaxed=True)
        # filter out evidences retrieved for the answer (noisy!)
        answering_evidences = [
            evidence
            for evidence in answering_evidences
            if not all([entity["id"] in answer_ids for entity in evidence["retrieved_for_entity"]])
        ]
        self.logger.debug(f"gold_answers: {answers}")
        self.logger.debug(f"answering_evidences: {answering_evidences}")
        self.logger.info(f"length of answering evidences: {len(answering_evidences)}")
        return answering_evidences

    def _get_answer_connecting_disambiguations(self, disambiguations, answering_evidences, answers):
        """
        Extract the relevant disambiguations using the answering facts.
        Returns disambiguation tuples, which have the following form:
        (kb_item_id, surface_forms, label).
        There are multiple surface forms, since the same KB item can potentially
        be disambiguated for several different question words.
        """
        # create dict from item_id to surface forms
        inverse_disambiguations = dict()
        for disambiguation in disambiguations:
            surface_form = disambiguation["question_word"]
            item_id = disambiguation["item"]["id"]
            item_label = disambiguation["item"]["label"]

            # skip disambiguations that are the answer
            if item_id in answers:
                continue

            # skip disambiguations for stopwords (noise!)
            surface_form_words = self.string_lib.get_question_words(surface_form, ner=None)
            if len(surface_form_words) == 0:
                continue

            # remember in dict
            if item_id in inverse_disambiguations:
                inverse_disambiguations[item_id].append(surface_form)
            else:
                inverse_disambiguations[item_id] = [surface_form]

        # create disambiguation tuples
        disambiguation_tuples = list()
        for evidence in answering_evidences:
            # get items that led to the answering fact coming into the context
            for item in evidence["wikidata_entities"]:
                item_id = item["id"]
                item_label = item["label"]
                if item_id in inverse_disambiguations and self._valid_item(item["id"]):
                    surface_forms = inverse_disambiguations[item_id]
                    if not (item_id, surface_forms, item_label) in disambiguation_tuples:
                        disambiguation_tuples.append((item_id, surface_forms, item_label))
        return disambiguation_tuples

    def _valid_item(self, item):
        """
        Verify that the item is valid.
        1.  Checks whether the item is very frequent. For frequent items,
            the occurence in an extracted fact could be misleading.
        2.  Checks whether item is a predicate (predicates go into relation slot, not entity slot)
        """
        if item[0] == "P":  # predicates are dropped
            return False
        if self._item_is_country(item):  # countries are always frequent
            return False
        return self._freq_check(item)

    def _freq_check(self, item):
        try:
            freq1, freq2 = self.clocq.get_frequency(item)
            freq = freq1 + freq2
            return freq < 1000000
        except:
            time.sleep(3)
            self.logger.info(f"Wait to respond for the item: {item}")
            return self._freq_check(item)

    def _remove_surface_forms(self, question, disambiguation_tuples):
        """
        Remove disambiguated surface forms from question. Sort surface forms by
        length to avoid problems: e.g. removing 'unicorn' before removing 'last unicorn'
        leads to a problem.
        """
        # derive set of surface forms
        distinct_surface_forms = set()
        for (item_id, surface_forms, label) in disambiguation_tuples:
            distinct_surface_forms.update(surface_forms)
        # sort surface forms by string length
        distinct_surface_forms = sorted(distinct_surface_forms, key=lambda j: len(j), reverse=True)
        for surface_form in distinct_surface_forms:
            # mechanism to avoid lowering full question at this point
            start_index = question.lower().find(surface_form.lower())
            if not start_index == -1:
                end_index = start_index + len(surface_form)
                question = question[:start_index] + question[end_index:]
        return question

    def _get_answer_type(self, question, answers):
        """
        Get the answer_type from the answer.
        In case the answer has multiple types, compute the most relevant type to the question.
        """
        if self.string_lib.is_year(answers[0]["label"]):
            return "year"
        elif self.string_lib.is_timestamp(answers[0]["id"]):
            return "date"
        elif self.string_lib.is_number(answers[0]["id"]):
            return "number"
        elif self.string_lib.is_entity(answers[0]["id"]):
            type_ = self._get_most_relevant_type(answers)
            if type_ is None:
                return ""
            return type_["label"]
        else:
            return "string"

    def _type_relevance(self, type_id):
        """
        Score the relevance of the type.
        """
        if self.type_relevance_cache.get(type_id):
            return self.type_relevance_cache.get(type_id)
        freq1, freq2 = self.clocq.get_frequency(type_id)
        type_relevance = freq1 + freq2
        self.type_relevance_cache[type_id] = type_relevance
        return type_relevance

    def _get_most_relevant_type(self, answers):
        """
        Get the most relevant type for the item, as given by the type_relevance funtion.
        """
        # fetch types
        all_types = list()
        for item in answers:
            item_id = item["id"]
            types = self.clocq.get_types(item_id)
            if not types:
                continue
            for type_ in types:
                if type_ != "None":
                    all_types.append(type_)
        if not all_types:
            return None
        # sort types by relevance, and take top one
        most_relevant_type = sorted(
            all_types, key=lambda j: self._type_relevance(j["id"]), reverse=True
        )[0]
        return most_relevant_type

    def _normalize_relation_tsf(self, relation_tsf):
        """Remove punctuation, whitespaces and lower the string."""
        relation_tsf = (
            relation_tsf.replace(",", "")
                .replace("!", "")
                .replace("?", "")
                .replace(".", "")
                .replace("'", "")
                .replace('"', "")
                .replace(":", "")
                .replace("’", "")
                .replace("{", "")
                .replace("}", "")
                .replace(" s ", " ")
        )
        while "  " in relation_tsf:
            relation_tsf = relation_tsf.replace("  ", " ")
        # relation_str = relation_str.lower()
        relation_tsf = relation_tsf.strip()
        return relation_tsf

    def _item_is_country(self, item_id):
        """
        Check if the item is of type country.
        """
        if item_id[0] != "Q":
            return False

        try:
            types = self.clocq.get_types(item_id)

            if not types or types == ["None"]:
                return False
            type_ids = [type_["id"] for type_ in types]
            if "Q6256" in type_ids:  # country type
                return True
        except:
            time.sleep(3)
            self.logger.info(f"Wait to respond for the item: {item_id}")
            return self._item_is_country(item_id)

