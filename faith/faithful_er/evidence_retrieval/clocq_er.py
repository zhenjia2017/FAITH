import os
import re
import copy
import pickle
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from clocq.CLOCQ import CLOCQ
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient

from faith.library.utils import get_logger, get_config
from faith.faithful_er.evidence_retrieval.wikipedia_retriever.wikipedia_retriever import WikipediaRetriever

ENT_PATTERN = re.compile("^Q[0-9]+$")
PRE_PATTERN = re.compile("^P[0-9]+$")

KB_ITEM_SEPARATOR = ", "


class ClocqRetriever:
    def __init__(self, config, temporal_value_annotator):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.temporal_value_annotator = temporal_value_annotator
        self.library = self.temporal_value_annotator.library
        # load cache
        self.use_cache = config["er_use_cache"]

        if self.use_cache:
            self.cache_path = os.path.join(config["path_to_data"], config["benchmark"], config["er_cache_path"])
            self._init_cache()
            self.cache_changed = False

        if config["clocq_use_api"]:
            self.clocq = CLOCQInterfaceClient(host=config["clocq_host"], port=config["clocq_port"])
        else:
            self.clocq = CLOCQ()

        # initialize wikipedia-retriever
        self.wiki_retriever = WikipediaRetriever(config, self.temporal_value_annotator)

    def retrieve_evidences(self, query, sources):
        """
        Retrieve evidences and question entities
        for the given query.

        This function is used for initial evidence
        retrieval. These evidences are filtered in the
        next step.

        Can also be used from external modules to access
        all evidences for the given TSF (if possible from cache).
        """
        self.logger.info(f"Retrieve evidences for: {query}")
        # first get question entities (and KB-facts if required)
        start = time.time()
        evidences, question_entities = self.retrieve_KB_facts(query)
        self.logger.info(f"Time taken (retrieve_KB_facts): {time.time() - start} seconds")
        self.logger.info(f"Number of question entities : {len(question_entities)}")

        # wikipedia evidences (only if required)
        if any(src in sources for src in ["text", "table", "info"]):
            start = time.time()
            wiki_evidences = self.retrieve_wikipedia_evidences_multithread(question_entities)

            for evi in wiki_evidences:
                evidences += evi

            self.logger.info(f"Time taken (retrieve_wikipedia_evidences): {time.time() - start} seconds")

        # remove duplicated evidences
        evidences = self.remove_duplicate_evidence(evidences)

        # config-based filtering
        evidences = self.filter_evidences(evidences, sources)
        self.logger.info(f"Number of evidences : {len(evidences)}")
        return evidences, question_entities

    def retrieve_wikipedia_evidences(self, question_entity):
        """
        Retrieve evidences from Wikipedia for the given question entity.
        """

        question_entity = question_entity["item"]

        # retrieve result
        evidences = self.wiki_retriever.retrieve_wp_evidences(question_entity)

        assert not evidences is None  # evidences should never be None

        return evidences

    def _kb_fact_to_evidence_multithread(self, facts, question_items_set):
        with ThreadPoolExecutor(max_workers=5) as executor:
            evidences = [
                future.result()
                for future in [
                    executor.submit(self._kb_fact_to_evidence, fact, question_items_set)
                    for fact in facts
                ]
            ]
            return evidences

    def retrieve_wikipedia_evidences_multithread(self, question_entitiess):
        with ThreadPoolExecutor(max_workers=5) as executor:
            evidences = [
                future.result()
                for future in [
                    executor.submit(self.retrieve_wikipedia_evidences, question_entity)
                    for question_entity in question_entitiess
                ]
            ]
            return evidences

    def retrieve_KB_facts(self, tsf, recursive_calls=0):
        """
        Retrieve KB facts for the given tsf (or other question/text).
        Also returns the question entities, for usage in Wikipedia retriever.
        """
        # look-up cache for kb fact

        if recursive_calls > 5:
            return []

        if self.use_cache and tsf in self.cache:
            clocq_result = copy.deepcopy(self.cache[tsf])
            self.logger.debug(f"Have cache hit: Retrieving search space for: {tsf}.")

        else:
            self.logger.debug(f"No cache hit: Retrieving search space for: {tsf}.")

            # apply CLOCQ
            start = time.time()
            try:
                clocq_result = self.clocq.get_search_space(tsf, parameters=self.config["clocq_params"],
                                                           include_labels=True,
                                                           include_type=True)
                self.logger.info(f"Time taken (clocq.get_search_space): {time.time() - start} seconds")
                # store result in cache

            except:
                time.sleep(1)
                return self.retrieve_KB_facts(tsf, recursive_calls + 1)

            if self.use_cache:
                # if self.use_cache and self.config.get("ers_update_clocq_cache", True):
                self.cache_changed = True
                self.cache[tsf] = copy.deepcopy(clocq_result)

        # get question entities (predicates dropped)
        question_entities = [
            item
            for item in clocq_result["kb_item_tuple"]
            if not item["item"]["id"] is None and ENT_PATTERN.match(item["item"]["id"])
        ]

        question_items_set = set([item["item"]["id"] for item in clocq_result["kb_item_tuple"]])

        evidences = self._kb_fact_to_evidence_multithread(
            clocq_result["search_space"], question_items_set
        )

        return evidences, question_entities

    def remove_duplicate_evidence(self, evidences):
        """
        evidence = {
                "evidence_text": evidence_text,
                "wikidata_entities": wikidata_entities,
                "disambiguations": [
                        (item["label"], item["id"]) for item in kb_fact if ENT_PATTERN.match(item["id"])
                ],
                "retrieved_for_entity": retrieved_for,
                "tempinfo": [timespan, disambiguation] if timespan and disambiguation else None,
                "source": "kb",
        }
        """
        evi_dic = {}
        for evidence in evidences:
            text = evidence["evidence_text"]
            source = evidence["source"]
            # keep unique evidence text per source
            text_source = f"{text}|||{source}"
            if text_source not in evi_dic:
                evi_dic[text_source] = {
                    "wikidata_entities": [],
                    "disambiguations": [],
                    "retrieved_for_entity": [],
                    "tempinfo": evidence["tempinfo"],
                }

            for item in evidence["wikidata_entities"]:
                if item not in evi_dic[text_source]["wikidata_entities"]:
                    evi_dic[text_source]["wikidata_entities"].append(item)

            for item in evidence["disambiguations"]:
                if item not in evi_dic[text_source]["disambiguations"]:
                    evi_dic[text_source]["disambiguations"].append(item)

            for item in evidence["retrieved_for_entity"]:
                if item not in evi_dic[text_source]["retrieved_for_entity"]:
                    evi_dic[text_source]["retrieved_for_entity"].append(item)

        for key, value in evi_dic.items():
            text = key.split("|||")[0]
            source = key.split("|||")[1]
            value.update({"evidence_text": text, "source": source})

        return list(evi_dic.values())

    def _kb_fact_to_evidence(self, kb_fact, question_items_set):
        """Transform the given KB-fact to an evidence."""

        def _format_fact(kb_fact):
            """Correct format of fact (if necessary)."""
            start_timestamp = None
            end_timestamp = None
            timespan = []
            disambiguation = list()
            retrieved_for = list()
            for item in kb_fact:
                index = kb_fact.index(item)
                item_pre = kb_fact[index - 1]
                if item["id"] in question_items_set:
                    retrieved_for.append(item)
                if self.library.is_timestamp(item["id"]):
                    item["label"] = self.library.convert_timestamp_to_date(item["id"])
                    item["label"] = item["label"].replace('"', "")
                    item["id"] = item["id"].replace('"', "")
                    if (item["label"], item["id"]) not in disambiguation:
                        disambiguation.append((item["label"], item["id"]))
                    if item_pre["id"] == 'P580':
                        # start time
                        start_timestamp = item["id"]
                    elif item_pre["id"] == 'P582':
                        # end time
                        if "-01-01T00:00:00Z" in item["id"]:
                            # timestamp is year, end time in timespan is changed to YYYY-12-31
                            end_timestamp = item["id"].replace("-01-01T00:00:00Z", "-12-31T00:00:00Z")
                        else:
                            end_timestamp = item["id"]
                    else:
                        # point in time or other time
                        if "-01-01T00:00:00Z" in item["id"]:
                            start_timestamp = item["id"]
                            # timestamp is year, end time in timespan is changed to YYYY-12-31
                            end_timestamp = item["id"].replace("-01-01T00:00:00Z", "-12-31T00:00:00Z")
                        else:
                            start_timestamp = item["id"]
                            end_timestamp = item["id"]

            # generate timespan with start time and end time
            if start_timestamp and end_timestamp:
                timespan.append([start_timestamp, end_timestamp])
            elif start_timestamp and not end_timestamp:
                # no end time
                timespan.append([start_timestamp, None])
            elif not start_timestamp and end_timestamp:
                # no start time
                timespan.append([None, end_timestamp])
            return kb_fact, timespan, disambiguation, retrieved_for

        def _get_wikidata_entities(kb_fact):
            """Return wikidata_entities for fact."""
            items = list()
            for item in kb_fact:
                # skip undesired answers
                if not _is_potential_answer(item["id"]):
                    continue
                # append to set
                items.append(item)
                # augment candidates with years (for differen granularity of answer)
                if self.library.is_timestamp(item["id"]):
                    year = self.library.get_year(item["id"])
                    new_item = {
                        "id": self.library.convert_year_to_timestamp(year),
                        "label": year,
                    }
                    items.append(new_item)
            return items

        def _is_potential_answer(item_id):
            """Return if item_id could be answer."""
            # keep all KB-items except for predicates
            if PRE_PATTERN.match(item_id):
                return False
            return True

        # evidence text
        kb_fact, timespan, disambiguation, retrieved_for = _format_fact(kb_fact)
        evidence_text = self._kb_fact_to_text(kb_fact)
        wikidata_entities = _get_wikidata_entities(kb_fact)

        # add tempinfo key to record temporal information of a fact
        evidence = {
            "evidence_text": evidence_text,
            "wikidata_entities": wikidata_entities,
            "disambiguations": [
                (item["label"], item["id"]) for item in kb_fact if ENT_PATTERN.match(item["id"])
            ],
            "retrieved_for_entity": retrieved_for,
            "tempinfo": [timespan, disambiguation] if timespan and disambiguation else None,
            "source": "kb",
        }
        return evidence

    def retrieve_evidences_from_kb(self, item):
        """Retrieve evidences from KB for the given item (used in DS)."""
        facts = self.clocq.get_neighborhood(
            item["id"], p=self.config["clocq_p"], include_labels=True
        )
        return [self._kb_fact_to_evidence(kb_fact, item) for kb_fact in facts]

    def filter_evidences(self, evidences, sources):
        """
        Filter the set of evidences according to their source.
        """
        filtered_evidences = list()
        for evidence in evidences:
            if len(evidence["wikidata_entities"]) == 1:
                continue
            if len(evidence["wikidata_entities"]) > self.config["evr_max_entities"]:
                continue
            if evidence["source"] in sources:
                filtered_evidences.append(evidence)
        return filtered_evidences

    def _kb_fact_to_text(self, kb_fact):
        """Verbalize the KB-fact."""
        return KB_ITEM_SEPARATOR.join([item["label"] for item in kb_fact])

    def store_cache(self):
        """Store the cache to disk."""
        if not self.use_cache:  # store only if cache in use
            return
        if not self.cache_changed:  # store only if cache changed
            return
        # check if the cache was updated by other processes
        if self._read_cache_version() == self.cache_version:
            # no updates: store and update version
            self.logger.info(f"Writing ER cache at path {self.cache_path}.")
            # with FileLock(f"{self.cache_path}.lock"):
            self._write_cache(self.cache)
            self._write_cache_version()
        else:
            # update! read updated version and merge the caches
            self.logger.info(f"Merging ER cache at path {self.cache_path}.")
            # with FileLock(f"{self.cache_path}.lock"):
            # read updated version
            updated_cache = self._read_cache()
            # overwrite with changes in current process (most recent)
            updated_cache.update(self.cache)
            # store
            self._write_cache(updated_cache)
            self._write_cache_version()
        # store extended wikipedia dump (if any changes occured)
        # self.wiki_retriever.store_dump()

    def reset_cache(self):
        """Reset the cache for new population."""
        self.logger.warn(f"Resetting ER cache at path {self.cache_path}.")
        # with FileLock(f"{self.cache_path}.lock"):
        self.cache = {}
        self._write_cache(self.cache)
        self._write_cache_version()

    def _init_cache(self):
        """Initialize the cache."""
        if os.path.isfile(self.cache_path):
            # remember version read initially
            self.logger.info(f"Loading ER cache from path {self.cache_path}.")
            # with FileLock(f"{self.cache_path}.lock"):
            self.cache_version = self._read_cache_version()
            self.logger.debug(self.cache_version)
            self.cache = self._read_cache()
            self.logger.info(f"ER cache successfully loaded.")
        else:
            self.logger.info(f"Could not find an existing ER cache at path {self.cache_path}.")
            self.logger.info("Populating ER cache from scratch!")
            self.cache = {}
            self._write_cache(self.cache)
            self._write_cache_version()

    def _read_cache(self):
        """
        Read the current version of the cache.
        This can be different from the version used in this file,
        given that multiple processes may access it simultaneously.
        """
        # read file content from cache shared across QU methods
        with open(self.cache_path, "rb") as fp:
            self.logger.info("Opened cache file. Starting to load data.")
            cache = pickle.load(fp)
            self.logger.info("Done loading the data.")
        return cache

    def _write_cache(self, cache):
        """Write to the cache."""
        cache_dir = os.path.dirname(self.cache_path)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as fp:
            pickle.dump(cache, fp)
        return cache

    def _read_cache_version(self):
        """Read the cache version (hashed timestamp of last update) from a dedicated file."""
        if not os.path.isfile(f"{self.cache_path}.version"):
            self._write_cache_version()
        with open(f"{self.cache_path}.version", "r") as fp:
            cache_version = fp.readline().strip()
        return cache_version

    def _write_cache_version(self):
        """Write the current cache version (hashed timestamp of current update)."""
        with open(f"{self.cache_path}.version", "w") as fp:
            version = str(time.time())
            fp.write(version)
        self.cache_version = version
