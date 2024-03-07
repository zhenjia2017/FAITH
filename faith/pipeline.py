import os
import sys
import json
import time
import copy
import logging
from faith.library.utils import get_config, get_logger, get_result_logger, store_json_with_mkdir
# tqu
from faith.temporal_qu.seq2seq_tqu_iques import Seq2SeqIQUESTQU
# fer
from faith.faithful_er.fer import FER
# ha
from faith.heterogeneous_answering.graph_neural_network.iterative_gnns import IterativeGNNs
from faith.heterogeneous_answering.seq2seq_answering.seq2seq_answering_module import Seq2SeqAnsweringModule

class Pipeline:
    def __init__(self, config):
        """Create the pipeline based on the config."""
        # load config
        self.config = config
        self.logger = get_logger(__name__, config)
        self.result_logger = get_result_logger(config)

        # load individual modules
        self.fer = self._load_fer()
        self.ha = self._load_ha()
        self.tqu = self._load_tqu()
        self.name = self.config["name"]
        self.benchmark = self.config["benchmark"]
        self.faith = self.config["faith_or_unfaith"]

        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        print("Loggers", loggers)

    def train(self, sources_str, modules_to_train=("tqu", "fer", "ha")):
        """
        Please set run_tvr as False when training the models
        Train the given pipeline in the standard manner.
        """
        sources = sources_str.split("_")

        # Temporal Question Understanding (TQU)
        if "tqu" in modules_to_train:
            # Step1: Fine-tune Seq2seq model (BART) in TQU
            # Before fine-tuning Seq2seq model, please firstly generate annotated TSFs via distant supervision.
            step1_start = time.time()
            self.logger.info(f"Step1: Start training TQU seq2seq model for generating TSFs")
            self.tqu.train()
            self.logger.info(f"Time taken (Training TQU Model): {time.time() - step1_start} seconds")

            # Step2: Inference TSFs for train and dev sets using the fine-tuned Seq2seq model
            #        This step is required in both seq2seq and without TQU settings
            step2_start = time.time()
            self.logger.info(f"Step2: Start inference TSFs for train and dev sets")
            self.tqu.inference()
            self.logger.info(f"Time taken (Inference TSFs): {time.time() - step2_start} seconds")
            self.tqu = None  # free up memory

        # Faithful Evidence Retrieval and Scoring (ERS)
        if "fer" in modules_to_train:
            # Step3: Evidence Retrieval
            step3_start = time.time()
            self.logger.info(f"Step3: Start retrieving evidences of TSFs for train and dev sets")
            # self.fer.er_inference(sources)
            # # store results in cache
            # self.fer.store_cache()
            # self.logger.info(f"Time taken (Evidence Retrieval): {time.time() - step3_start} seconds")
            #
            # # Step4: Evidence Pruning
            # if self.config["faith_or_unfaith"] == "faith":
            #     step4_start = time.time()
            #     self.logger.info(
            #         f"Step4: Start pruning evidences for train and dev sets")
            #     self.prune()
            #     self.logger.info(f"Time taken (Evidence Pruning): {time.time() - step4_start} seconds")

            # Step5: Train evidence scoring model based on SBERT
            step5_start = time.time()
            self.logger.info(f"Step5: Start training evidence scoring model")
            self.fer.train()
            self.logger.info(f"Time taken (Train Evidence Scoring Model): {time.time() - step5_start} seconds")

            # Step6: Scoring evidence and select top-100 evidence as the input for training HA model
            step6_start = time.time()
            self.logger.info(
                f"Step6: Start scoring evidence")
            self.fer.evs_inference()
            self.logger.info(f"Time taken (Scoring Evidence): {time.time() - step6_start} seconds")
            self.fer = None  # free up memory

        # Explainable Heterogeneous Answering (HA)
        if "ha" in modules_to_train:
            # Step7: Train HA model
            step7_start = time.time()
            self.logger.info(f"Step7: Start train HA model")
            self.ha.train(sources)
            self.logger.info(f"Time taken (Training HA model): {time.time() - step7_start} seconds")

    def source_combinations(self, dev=False):
        """
        Run the pipeline using gold answers in the WWW 2024 paper.
        """
        source_combinations = [
            "kb_text_table_info",
            "kb",
            "text",
            "table",
            "info",
            "kb_text",
            "kb_table",
            "kb_info",
            "text_table",
            "text_info",
            "table_info",
        ]
        self.evaluate(dev=dev, clean_up=False, sources_str=source_combinations)

    def top_answer_for_tvr(self, top_answers, dev=False):
        """
        Run the pipeline using gold answers in the WWW 2024 paper.
        """
        self.evaluate_tvr(dev=dev, clean_up=False, top_answers=top_answers)

    def evaluate_tvr(self, dev=False, clean_up=False, top_answers=[1], sources_str="kb_text_table_info"):
        # data path
        benchmark_path = self.config["benchmark_path"]
        input_dir = os.path.join(benchmark_path, self.benchmark)
        evs_max_evidences = self.config["evs_max_evidences"]
        # either use given option, or from config
        ha = self.config["ha"]
        tqu_oracle_temporal_category = "no-oracle-category"
        tqu_oracle_temporal_value = "no-oracle-value"
        if "tqu_oracle_temporal_category" in self.config and self.config["tqu_oracle_temporal_category"]:
            tqu_oracle_temporal_category = "oracle-category"
        if "tqu_oracle_temporal_value" in self.config and self.config["tqu_oracle_temporal_value"]:
            tqu_oracle_temporal_value = "oracle-value"

        run_tvr = "tvr"
        if not self.config["run_tvr"]:
            run_tvr = "no_tvr"
        # either use given option, or from config
        ha = self.config["ha"]
        if ha == "seq2seq_ha":
            gnn_max_evidences = ''
        else:
            gnn_max_evidences = []
            for i in range(len(self.config["gnn_inference"])):
                gnn_max_evidences.append(str(self.config["gnn_inference"][i]["gnn_max_evidences"]))

            gnn_max_evidences = '_'.join(gnn_max_evidences)

        for tvr_top_answer in top_answers:
            if dev:
                input_path = os.path.join(input_dir, self.config["dev_input_path"])
            else:
                input_path = os.path.join(input_dir, self.config["test_input_path"])

            with open(input_path, "r") as fp:
                data = json.load(fp)

            # run inference on data
            sources = sources_str.split("_")
            # define output path
            output_dir = self.set_output_dir(sources_str)
            # tqu inference
            self.tqu.inference_on_data(data, tvr_top_answer, sources)

            if clean_up:
                self.tqu = None  # free up memory
            if dev:
                output_path = f"{output_dir}/{ha}_{gnn_max_evidences}_res_dev_{self.faith}_{tqu_oracle_temporal_category}_{tqu_oracle_temporal_value}_{run_tvr}_e{evs_max_evidences}_t{tvr_top_answer}_tqu.json"
            else:
                output_path = f"{output_dir}/{ha}_{gnn_max_evidences}_res_test_{self.faith}_{tqu_oracle_temporal_category}_{tqu_oracle_temporal_value}_{run_tvr}_e{evs_max_evidences}_t{tvr_top_answer}_tqu.json"
            store_json_with_mkdir(data, output_path)

            with open(output_path, "r") as fp:
                data = json.load(fp)

            input_data = copy.deepcopy(data)

            self.fer.inference_on_data(input_data, sources)

            if dev:
                fer_output_path = f"{output_dir}/{ha}_{gnn_max_evidences}_res_dev_{self.faith}_{tqu_oracle_temporal_category}_{tqu_oracle_temporal_value}_{run_tvr}_e{evs_max_evidences}_t{tvr_top_answer}_ers.jsonl"
            else:
                fer_output_path = f"{output_dir}/{ha}_{gnn_max_evidences}_res_test_{self.faith}_{tqu_oracle_temporal_category}_{tqu_oracle_temporal_value}_{run_tvr}_e{evs_max_evidences}_t{tvr_top_answer}_ers.jsonl"

            store_json_with_mkdir(input_data, fer_output_path)

            with open(fer_output_path, "r") as fp:
                input_data = json.load(fp)
            self.ha.inference_on_data(input_data, sources)
            if dev:
                ha_output_path = f"{output_dir}/{ha}_{gnn_max_evidences}_res_dev_{self.faith}_{tqu_oracle_temporal_category}_{tqu_oracle_temporal_value}_{run_tvr}_e{evs_max_evidences}_t{tvr_top_answer}_gold_answers.json"
            else:
                ha_output_path = f"{output_dir}/{ha}_{gnn_max_evidences}_res_test_{self.faith}_{tqu_oracle_temporal_category}_{tqu_oracle_temporal_value}_{run_tvr}_e{evs_max_evidences}_t{tvr_top_answer}_gold_answers.json"

            store_json_with_mkdir(input_data, ha_output_path)

            # evaluate performance of ha results
            self.compute_ha_metrics(ha_output_path, sources_str)
        # store the cache
        self.fer.store_cache()

    def compute_ha_metrics(self, ha_output_path, sources_str="kb_text_table_info"):
        # compute results
        self.ha.evaluate_hs_results(ha_output_path, sources_str)

    def compute_fer_metrics(self, fer_output_path, stage="scoring"):
        self.fer.evaluate_retrieval_results_res_stage(fer_output_path, stage=stage)

    def evaluate(self, dev=False, clean_up=False, sources_str="kb_text_table_info"):
        # define output path
        if not isinstance(sources_str, list):
            source_combinations = [sources_str]
        else:
            source_combinations = sources_str

        # data path
        benchmark_path = self.config["benchmark_path"]
        input_dir = os.path.join(benchmark_path, self.benchmark)
        evs_max_evidences = self.config["evs_max_evidences"]
        # either use given option, or from config
        tqu_oracle_temporal_category = "no-oracle-category"
        tqu_oracle_temporal_value = "no-oracle-value"
        if "tqu_oracle_temporal_category" in self.config and self.config["tqu_oracle_temporal_category"]:
            tqu_oracle_temporal_category = "oracle-category"
        if "tqu_oracle_temporal_value" in self.config and self.config["tqu_oracle_temporal_value"]:
            tqu_oracle_temporal_value = "oracle-value"
        run_tvr = "tvr"
        if not self.config["run_tvr"]:
            run_tvr = "no_tvr"
        # either use given option, or from config
        ha = self.config["ha"]
        if ha == "seq2seq_ha":
            gnn_max_evidences = ''
        else:
            gnn_max_evidences = []
            for i in range(len(self.config["gnn_inference"])):
                gnn_max_evidences.append(str(self.config["gnn_inference"][i]["gnn_max_evidences"]))

            gnn_max_evidences = '_'.join(gnn_max_evidences)

        if "tvr_topk_answer" in self.config:
            tvr_top_answer = self.config["tvr_topk_answer"]
        else:
            tvr_top_answer = 1
        if dev:
            input_path = os.path.join(input_dir, self.config["dev_input_path"])
        else:
            input_path = os.path.join(input_dir, self.config["test_input_path"])

        with open(input_path, "r") as fp:
            data = json.load(fp)

        # run inference on data
        for sources_str in source_combinations:
            sources = sources_str.split("_")
            # define output path
            output_dir = self.set_output_dir(sources_str)
            # tqu inference
            self.tqu.inference_on_data(data, tvr_top_answer, sources)

            if clean_up:
                self.tqu = None  # free up memory
            if dev:
                output_path = f"{output_dir}/{ha}_{gnn_max_evidences}_res_dev_{self.faith}_{tqu_oracle_temporal_category}_{tqu_oracle_temporal_value}_{run_tvr}_e{evs_max_evidences}_t{tvr_top_answer}_tqu.json"
            else:
                output_path = f"{output_dir}/{ha}_{gnn_max_evidences}_res_test_{self.faith}_{tqu_oracle_temporal_category}_{tqu_oracle_temporal_value}_{run_tvr}_e{evs_max_evidences}_t{tvr_top_answer}_tqu.json"
            store_json_with_mkdir(data, output_path)

            with open(output_path, "r") as fp:
                data = json.load(fp)

            input_data = copy.deepcopy(data)

            self.fer.inference_on_data(input_data, sources)

            if dev:
                fer_output_path = f"{output_dir}/{ha}_{gnn_max_evidences}_res_dev_{self.faith}_{tqu_oracle_temporal_category}_{tqu_oracle_temporal_value}_{run_tvr}_e{evs_max_evidences}_t{tvr_top_answer}_ers.jsonl"
            else:
                fer_output_path = f"{output_dir}/{ha}_{gnn_max_evidences}_res_test_{self.faith}_{tqu_oracle_temporal_category}_{tqu_oracle_temporal_value}_{run_tvr}_e{evs_max_evidences}_t{tvr_top_answer}_ers.jsonl"

            store_json_with_mkdir(input_data, fer_output_path)

            self.fer.store_cache()
            if clean_up:
                self.fer = None  # free up memory

            with open(fer_output_path, "r") as fp:
                input_data = json.load(fp)
            self.ha.inference_on_data(input_data, sources)
            if dev:
                ha_output_path = f"{output_dir}/{ha}_{gnn_max_evidences}_res_dev_{self.faith}_{tqu_oracle_temporal_category}_{tqu_oracle_temporal_value}_{run_tvr}_e{evs_max_evidences}_t{tvr_top_answer}_gold_answers.json"
            else:
                ha_output_path = f"{output_dir}/{ha}_{gnn_max_evidences}_res_test_{self.faith}_{tqu_oracle_temporal_category}_{tqu_oracle_temporal_value}_{run_tvr}_e{evs_max_evidences}_t{tvr_top_answer}_gold_answers.json"

            store_json_with_mkdir(input_data, ha_output_path)

            # compute answer presence of fer results
            self.compute_fer_metrics(fer_output_path, "initial")
            self.compute_fer_metrics(fer_output_path, "pruning")
            self.compute_fer_metrics(fer_output_path, "scoring")
            # evaluate performance of ha results
            self.compute_ha_metrics(ha_output_path, sources_str)

    def compute_metrics(self, input_data, sources_str):
        # compute results
        p_at_1_list = [instance["p_at_1"] for instance in input_data]
        p_at_1 = sum(p_at_1_list) / len(p_at_1_list)
        p_at_1 = round(p_at_1, 3)
        num_questions = len(p_at_1_list)
        # log result
        res_str = f"Gold answers - {sources_str} - P@1 ({num_questions}): {p_at_1}"
        self.logger.info(res_str)
        self.result_logger.info(res_str)

        # compute results
        mrr_list = [instance["mrr"] for instance in input_data]
        mrr = sum(mrr_list) / len(mrr_list)
        mrr = round(mrr, 3)
        num_questions = len(mrr_list)
        # log result
        res_str = f"Gold answers - {sources_str} - MRR ({num_questions}): {mrr}"
        self.logger.info(res_str)
        self.result_logger.info(res_str)

        # compute results
        hit_at_5_list = [instance["h_at_5"] for instance in input_data]
        hit_at_5 = sum(hit_at_5_list) / len(hit_at_5_list)
        hit_at_5 = round(hit_at_5, 3)
        # log result
        res_str = f"Gold answers - {sources_str} - H@5 ({num_questions}): {hit_at_5}"
        self.logger.info(res_str)
        self.result_logger.info(res_str)

    def prune(self, sources_str="kb_text_table_info"):
        benchmark = self.config["benchmark"]
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]
        tqu = self.config["tqu"]
        method_name = self.config["name"]
        fer = self.config["fer"]
        sources = sources_str.split("_")
        start = time.time()
        input_path = os.path.join(input_dir, benchmark, tqu, fer, sources_str, f"train_er-{method_name}.jsonl")
        output_path = os.path.join(output_dir, benchmark, tqu, fer, sources_str, f"train_erp-{method_name}.jsonl")
        self.fer.prune_on_data_split(input_path, output_path, sources)
        self.fer.evaluate_retrieval_results(output_path)
        running_time = time.time() - start
        self.logger.info(f"Pruning time for the train set is {running_time}")

        start = time.time()
        input_path = os.path.join(input_dir, benchmark, tqu, fer, sources_str, f"dev_er-{method_name}.jsonl")
        output_path = os.path.join(output_dir, benchmark, tqu, fer, sources_str, f"dev_erp-{method_name}.jsonl")
        self.fer.prune_on_data_split(input_path, output_path, sources)
        self.fer.evaluate_retrieval_results(output_path)
        running_time = time.time() - start
        self.logger.info(f"Pruning time for the dev set is {running_time}")

    def example(self, sources_str):
        """Run pipeline on a single input question."""
        instance = {
            "Id": 8111,
            "Question creation date": "2023-07-15",
            "Question": "What national rugby union team did Gary Seear play on when he was a member of the Fracasso San Dona club?",
            "answers": [{"id": "Q55801", "label": "New Zealand national rugby union team"}]}
        topk_answers = 1
        # run inference
        sources = sources_str.split("_")
        start = time.time()
        instance = self.inference_on_instance(instance, topk_answers, sources)
        self.logger.info(instance)
        self.logger.info(f"Time taken (ALL): {time.time() - start} seconds")

    def inference_on_instance(self, instance, topk_answers, sources=["kb", "text", "table", "info"]):
        """Run pipeline on given instance."""
        start = time.time()
        self.tqu.inference_on_instance(instance, topk_answers, sources)
        self.logger.info(instance)
        self.logger.info(f"Time taken (TQU): {time.time() - start} seconds")
        self.logger.info(f"Running FER")
        self.fer.inference_on_instance(instance, sources)
        self.logger.info(instance)
        self.logger.info(f"Time taken (TQU, FER): {time.time() - start} seconds")
        self.logger.info(f"Running HA")
        self.ha.inference_on_instance(instance, sources)
        self.logger.info(instance)
        self.logger.info(f"Time taken (ALL): {time.time() - start} seconds")
        return instance

    def set_output_dir(self, sources_str):
        """Define path for outputs."""
        tqu = self.config["tqu"]
        fer = self.config["fer"]
        ha = self.config["ha"]
        path_to_intermediate_results = self.config["path_to_intermediate_results"]
        output_dir = os.path.join(path_to_intermediate_results, self.benchmark, tqu, fer, sources_str, ha)
        return output_dir

    def _load_tqu(self):
        """Instantiate TQU stage of FAITH pipeline."""
        tqu = self.config["tqu"]
        self.logger.info("Loading TQU module")
        if tqu == "seq2seq_tqu":
            return Seq2SeqIQUESTQU(self.config, self)
        else:
            raise ValueError(
                f"There is no available module for instantiating the TQU phase called {tqu}."
            )

    def _load_fer(self):
        """Instantiate FER stage of FAITH pipeline."""
        fer = self.config["fer"]
        self.logger.info("Loading FER module")
        if fer == "fer":
            return FER(self.config)
        else:
            raise ValueError(
                f"There is no available module for instantiating the ERS phase called {fer}."
            )

    def _load_ha(self):
        """Instantiate HA stage of FAITH pipeline."""
        ha = self.config["ha"]
        self.logger.info("Loading HA module")
        if ha == "explaignn":
            return IterativeGNNs(self.config)
        elif ha == "seq2seq_ha":
            return Seq2SeqAnsweringModule(self.config)
        else:
            raise ValueError(
                f"There is no available module for instantiating the HA phase called {ha}."
            )


#######################################################################################################################
#######################################################################################################################
def main():
    # check if provided options are valid
    if len(sys.argv) < 3:
        raise Exception(
            "Usage: python faith/pipeline.py <FUNCTION> <PATH_TO_CONFIG> [<SOURCES_STRING>]"
        )

    # load config
    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    # train the models in tqu, fer and ha stages
    if function.startswith("--train_"):
        sources_str = sys.argv[3] if len(sys.argv) > 3 else "kb_text_table_info"
        pipeline = Pipeline(config)
        modules_to_train_str = function.replace("--train_", "")
        modules_to_train = ("tqu", "fer", "ha") if not modules_to_train_str else modules_to_train_str
        pipeline.train(sources_str, modules_to_train)

    elif function == "--source-combinations-test":
        pipeline = Pipeline(config)
        pipeline.source_combinations()

    elif function == "--source-combinations-dev":
        pipeline = Pipeline(config)
        pipeline.source_combinations(dev=True)

    elif function == "--top-answer-dev-3-5":
        pipeline = Pipeline(config)
        top_answers = [3,5]
        pipeline.top_answer_for_tvr(top_answers, dev=True)

    elif function == "--top-answer-test-3-5":
        pipeline = Pipeline(config)
        top_answers = [3, 5]
        pipeline.top_answer_for_tvr(top_answers)

    elif function == "--evaluate":
        sources_str = sys.argv[3] if len(sys.argv) > 3 else "kb_text_table_info"
        pipeline = Pipeline(config)
        pipeline.evaluate(sources_str=sources_str)

    elif function == "--evaluate-dev":
        sources_str = sys.argv[3] if len(sys.argv) > 3 else "kb_text_table_info"
        pipeline = Pipeline(config)
        pipeline.evaluate(dev=True, sources_str=sources_str)

    elif function == "--example":
        sources_str = sys.argv[3] if len(sys.argv) > 3 else "kb_text_table_info"
        pipeline = Pipeline(config)
        pipeline.example(sources_str)

    else:
        raise Exception(f"Unknown function {function}!")


if __name__ == "__main__":
    main()
