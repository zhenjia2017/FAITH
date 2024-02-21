import time
from faith.library.utils import get_logger
from faith.library.string_library import StringLibrary
from faith.faithful_er.evidence_retrieval.clocq_er import ClocqRetriever
from faith.faithful_er.evidence_pruning.pruning import EvidencePruning
from faith.faithful_er.faithful_evidence_retrieval import FaithfulEvidenceRetrieval
from faith.faithful_er.evidence_scoring.es_module import ESModule
from faith.library.temporal_library import TemporalValueAnnotator
from faith.evaluation import answer_presence


class FER(FaithfulEvidenceRetrieval):
    """
    Variant of the FER, which prunes and scores.
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.string_lib = StringLibrary(config)
        self.temporal_value_annotator = TemporalValueAnnotator(config, self.string_lib)
        self.evr = ClocqRetriever(config, self.temporal_value_annotator)
        self.evp = EvidencePruning(config)
        self.evs = ESModule(config)
        self.faith_or_unfaith = self.config["faith_or_unfaith"]
        self.max_evidence = self.config["evs_max_evidences"]

    def inference_on_instance(self, instance, sources=["kb", "text", "table", "info"]):
        """Retrieve candidate and prune for generating faithful evidences for TSF."""
        start = time.time()
        self.logger.debug(f"Running ER")
        tsf = instance["structured_temporal_form"]
        # tsf is a dictionary
        if isinstance(tsf, dict):
            if "entity" in tsf and "relation" in tsf and "answer_type" in tsf:
                entity = tsf["entity"].strip()
                relation = tsf["relation"].strip()
                answer_type = tsf["answer_type"].strip()
                # for keeping the consistency with the paper, we add answer type for retrieval
                query = f"{entity}{' '}{relation}{' '}{answer_type}"
        else:
            # when without TQU, the input is question itself
            query = tsf

        initial_evidences, question_entities = self.evr.retrieve_evidences(query, sources)
        instance["candidate_evidences"] = initial_evidences
        instance["question_entities"] = question_entities
        self.logger.debug(f"Time taken (ER): {time.time() - start} seconds")
        if "answers" not in instance:
            instance["answers"] = self.string_lib.format_answers(instance)
        initial_hit, initial_answering_evidences = answer_presence(initial_evidences, instance["answers"])
        instance["answer_presence_initial"] = initial_hit
        instance["answer_presence_per_src_initial"] = {
            evidence["source"]: 1 for evidence in initial_answering_evidences
        }

        if self.faith_or_unfaith == "faith":
            pruned_evidences = self.evp.pruning_evidences(tsf, initial_evidences, sources)
            pruned_hit, pruned_answering_evidences = answer_presence(pruned_evidences, instance["answers"])
            instance["answer_presence_pruning"] = pruned_hit
            instance["answer_presence_per_src_pruning"] = {
                evidence["source"]: 1 for evidence in pruned_answering_evidences
            }
            instance["candidate_evidences"] = pruned_evidences
        self.logger.debug(f"Time taken (ER, EP): {time.time() - start} seconds")
        # store the evidences with faithful tag
        top_evidences = self.evs.get_top_evidences(query, instance["candidate_evidences"], self.max_evidence)
        instance["candidate_evidences"] = top_evidences
        hit, answering_evidences = answer_presence(top_evidences, instance["answers"])
        instance["answer_presence"] = hit
        instance["answer_presence_per_src"] = {
            evidence["source"]: 1 for evidence in answering_evidences
        }

    def train(self, sources=["kb", "text", "table", "info"]):
        self.evs.train(sources=["kb", "text", "table", "info"])

    def er_inference_on_instance(self, instance, sources=["kb", "text", "table", "info"]):
        start = time.time()
        self.logger.debug(f"Running ER")
        tsf = instance["structured_temporal_form"]
        # tsf is a dictionary
        if isinstance(tsf, dict):
            if "entity" in tsf and "relation" in tsf:
                entity = tsf["entity"].strip()
                relation = tsf["relation"].strip()
                answer_type = tsf["answer_type"].strip()
                # for keeping the consistency with the paper, we add answer type for retrieval
                query = f"{entity}{' '}{relation}{' '}{answer_type}"
        else:
            # when without TQU, the input is question itself
            query = tsf

        evidences, question_entities = self.evr.retrieve_evidences(query, sources)
        instance["candidate_evidences"] = evidences
        instance["question_entities"] = question_entities
        self.logger.debug(f"Time taken (ER): {time.time() - start} seconds")
        return evidences

    def evs_inference_on_instance(self, instance, max_evidence=100, sources=["kb", "text", "table", "info"]):
        tsf = instance["structured_temporal_form"]
        # tsf is a dictionary
        if isinstance(tsf, dict):
            if "entity" in tsf and "relation" in tsf:
                entity = tsf["entity"].strip()
                relation = tsf["relation"].strip()
                answer_type = tsf["answer_type"].strip()
                # for keeping the consistency with the paper, we add answer type for retrieval
                query = f"{entity}{' '}{relation}{' '}{answer_type}"
        else:
            # when without TQU, the input is question itself
            query = tsf
        evidences = instance["candidate_evidences"]
        top_evidences = self.evs.get_top_evidences(query, evidences, max_evidence)
        instance["candidate_evidences"] = top_evidences
        return top_evidences

    def prune_on_instance(self, instance, sources):
        evidences = self.evp.pruning_evidences(instance["structured_temporal_form"], instance["candidate_evidences"],
                                               sources)
        return evidences

    def store_cache(self):
        """Store cache of evidence retriever."""
        # We do not store cache of the TSFs
        # (because cache will become too large)
        # -> We still store the cache of the Wikipedia retriever
        self.evr.store_cache()
        self.evr.wiki_retriever.store_dump()
