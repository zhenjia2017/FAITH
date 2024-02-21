import os
import sys
import torch

from faith.library.utils import get_config, get_logger, get_result_logger
from faith.heterogeneous_answering.heterogeneous_answering import HeterogeneousAnswering
from faith.heterogeneous_answering.seq2seq_answering.seq2seq_answering_model import (
    Seq2SeqAnsweringModel,
)
import faith.heterogeneous_answering.seq2seq_answering.dataset_seq2seq_answering as dataset
import faith.evaluation as evaluation


class Seq2SeqAnsweringModule(HeterogeneousAnswering):
    def __init__(self, config):
        """Initialize SR module."""
        self.config = config
        self.logger = get_logger(__name__, config)
        self.result_logger = get_result_logger(config)
        self.method_name = config["name"]
        # create model
        self.ha_model = Seq2SeqAnsweringModel(config)
        self.model_loaded = False

    def train(self, sources = ["kb", "text", "table", "info"]):
        """Train the model on generated HA data (skip instances for which answer is not there)."""
        # train model
        self.logger.info(f"Starting training...")

        # input paths
        method_name = self.config["name"]
        sources_str = "_".join(sources)
        faith_or_unfaith = self.config["faith_or_unfaith"]
        benchmark = self.config["benchmark"]
        data_dir = self.config["path_to_intermediate_results"]
        tqu = self.config["tqu"]
        fer = self.config["fer"]
        evs_model = self.config["evs_model"]
        evs_max_evidences = self.config["evs_max_evidences"]
        # load data
        if faith_or_unfaith == "faith":
            train_file = f"train_erps-{method_name}-{evs_model}-{evs_max_evidences}.jsonl"
            dev_file = f"dev_erps-{method_name}-{evs_model}-{evs_max_evidences}.jsonl"
        else:
            train_file = f"train_ers-{method_name}-{evs_model}-{evs_max_evidences}.jsonl"
            dev_file = f"dev_ers-{method_name}-{evs_model}-{evs_max_evidences}.jsonl"

        train_path = os.path.join(data_dir, benchmark, tqu, fer, sources_str, train_file)
        dev_path = os.path.join(data_dir, benchmark, tqu, fer, sources_str, dev_file)

        self.logger.info(f"Using training data from {train_path}")
        self.logger.info(f"Using validation data from {dev_path}")

        self.logger.info(f"Starting training...")
        self.ha_model.train(train_path, dev_path)
        self.logger.info(f"Finished training.")

    def inference_on_instance(self, instance, sources = ["kb", "text", "table", "info"]):
        """Run inference on a single turn."""
        with torch.no_grad():
            # load HA model (if required)
            self._load()

            # prepare input
            input_text = dataset.input_to_text(instance)

            # run inference
            generated_answer = self.ha_model.inference(input_text)
            instance["generated_answer"] = generated_answer
            ranked_answers = self.get_ranked_answers(generated_answer, instance)
            instance["ranked_answers"] = [
                {"answer":{"id": ans["answer"]["id"], "label": ans["answer"]["label"]}, "rank": ans["rank"]}
                for ans in ranked_answers
            ]

            # eval
            if "answers" in instance:
                p_at_1 = evaluation.precision_at_1(ranked_answers, instance["answers"])
                instance["p_at_1"] = p_at_1
                mrr = evaluation.mrr_score(ranked_answers, instance["answers"])
                instance["mrr"] = mrr
                h_at_5 = evaluation.hit_at_5(ranked_answers, instance["answers"])
                instance["h_at_5"] = h_at_5

            # delete noise
            if instance.get("top_evidences"):
                del instance["top_evidences"]
            if instance.get("question_entities"):
                del instance["question_entities"]
            return instance

    def _load(self):
        """Load the HA model."""
        # only load if not already done so
        if not self.model_loaded:
            self.ha_model.load()
            self.ha_model.set_eval_mode()
            self.model_loaded = True

