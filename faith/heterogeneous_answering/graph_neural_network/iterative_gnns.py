import copy
import json
import numpy as np
import os
import sys
import time
import torch
import random

from faith.heterogeneous_answering.graph_neural_network.graph_neural_network import GNNModule
from faith.heterogeneous_answering.heterogeneous_answering import HeterogeneousAnswering
from faith.library.utils import get_config, get_logger, store_json_with_mkdir

SEED = 7
START_DATE = time.strftime("%y-%m-%d_%H-%M", time.localtime())


class IterativeGNNs(HeterogeneousAnswering):
    def __init__(self, config):
        super(IterativeGNNs, self).__init__(config)
        self.logger = get_logger(__name__, config)

        self.configs = []
        self.gnns = []
        self.model_loaded = False

        # reproducibility
        random_seed = SEED
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.logger.info(f"Config: {json.dumps(self.config, indent=4)}")

    def _prepare_config_for_iteration(self, config_at_i):
        """
        Construct a standard GNN config for the given iteration, from the multi GNN config.
        """
        config_for_iteration = self.config.copy()
        for key, value in config_at_i.items():
            config_for_iteration[key] = value
        return config_for_iteration

    def train(self, sources=["kb", "text", "table", "info"], random_seed=None):
        """
        Train the model on generated HA data (skip instances for which answer is not there).
        """
        self.logger.info(f"Loading data...")

        # initialize GNNs
        self.configs = [
            self._prepare_config_for_iteration(config_at_i)
            for config_at_i in self.config["gnn_train"]
        ]

        self.gnns = list()
        for i in range(len(self.config["gnn_train"])):
            if random_seed:
                # should ensure that sequential training results in same models as individual training
                # -> reset random states before updates
                random.seed(random_seed)
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)
            gnn = GNNModule(self.configs[i], iteration=i + 1)
            self.gnns.append(gnn)

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

        # train the specified GNN variants
        for i in range(len(self.configs)):
            if random_seed:
                # should ensure that sequential training results in same models as individual training
                # -> reset random states before updates
                random.seed(random_seed)
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)
            self.gnns[i].train(sources, train_path, dev_path)

        # done
        self.logger.info(f"Finished training.")

    # def inference(self, sources=("kb", "text", "table", "info")):
    #     """
    #     Run inference for config. Can be used to continue after ERS results.
    #     """
    #     self.load()
    #
    #     # open data
    #     method_name = self.config["name"]
    #     _, data_dir = self._get_data_dirs(sources)
    #     input_path = f"{data_dir}/res_{method_name}_ers.json"
    #     with open(input_path, "r") as fp:
    #         input_data = json.load(fp)
    #
    #     # inference
    #     self.inference_on_data(input_data, sources)
    #
    #     # store result
    #     output_path = f"{data_dir}/res_{method_name}_gold_answers.json"
    #     store_json_with_mkdir(input_data, output_path)
    #
    #     # log results
    #     self.log_results(input_data)
    #     self.logger.info(f"Finished inference.")

    def inference_on_data(self, data, sources=("kb", "text", "table", "info"), train=False):
        """Run inference on a set of turns."""
        self.load()

        start = time.time()
        self.logger.info(f"Started to measure time.")

        turns_before_iteration = list()
        for i in range(len(self.config["gnn_inference"])):
            turns_before_iteration.append(copy.deepcopy(data))

            # compute ans pres
            answer_presence_list = [instance["answer_presence"] for instance in data]
            answer_presence = sum(answer_presence_list) / len(answer_presence_list)
            answer_presence = round(answer_presence, 3)
            num_questions = len(answer_presence_list)
            res_str = f"Inference - Ans. pres. ({num_questions}): {answer_presence}"
            self.logger.info(res_str)

            self.gnns[i].inference_on_instances(data)

        # remember top evidences
        # for turn_idx, instance in enumerate(turns):
        #     # identify supporting evidences
        #     instance["supporting_evidences"] = self.get_supporting_evidences(instance)
        #     instance["answering_evidences"] = self.get_answering_evidences(
        #         instance, turn_idx, turns_before_iteration
        #     )

        self.logger.info(f"Time consumed (average per question): {(time.time()-start)/len(data)}")
        self._log_results(data, sources)
        return data

    def inference_on_instance(self, instance, sources=("kb", "text", "table", "info"), train=False):

        if not self.model_loaded:
            self.load()

        """Run inference on a single instance."""
        for i in range(len(self.config["gnn_inference"])):
            self.gnns[i].inference_on_instance(instance)
        return instance

    def dev(self, sources=("kb", "text", "table", "info")):
        """Evaluate the iterative GNN on the dev set."""
        self._eval(sources, "dev")

    def test(self, sources=("kb", "text", "table", "info")):
        """Evaluate the iterative GNN on the test set."""
        self._eval(sources, "test")

    def _eval(self, sources, split="dev"):
        """Evaluate the iterative GNN on the given split."""
        # set paths
        method_name = self.config["name"]
        evs_max_evidences = self.config["evs_max_evidences"]
        answer_weight = str(self.config["gnn_train"][-1]["gnn_multitask_answer_weight"]).replace(".","")
        input_dir, output_dir = self._get_data_dirs(sources)
        input_path = os.path.join(input_dir, f"{split}_ers-{method_name}-{evs_max_evidences}.jsonl")
        output_path = os.path.join(output_dir, f"{split}_ha-{method_name}-{evs_max_evidences}-a{answer_weight}.json")

        # evaluate
        self.load()
        self.inference_on_data_split(input_path, output_path, sources, jsonl=False)
        self.logger.info(f"Finished evaluation on {split}-set.")

    def get_supporting_evidences(self, turn):
        """
        Get the supporting evidences for the answer.
        This function overwrites the model-agnostic implementation in
        the HeterogeneousAnswering class. The (top) evidences used in the
        final GNN layer are used as supporting evidences.
        """
        return turn["candidate_evidences"]

    def get_answering_evidences(self, turn, turn_idx, turns_before_iteration):
        """
        Get the neighboring evidences of the answer in the initial graph,
        i.e. the answering evidences.
        """
        num_explaining_evidences = self.config["ha_max_supporting_evidences"]
        top_evidences = turns_before_iteration[0][turn_idx]["candidate_evidences"]
        if not turn["ranked_answers"]:
            return []
        answer_entity = turn["ranked_answers"][0]["answer"]
        answering_evidences = [
            ev
            for ev in top_evidences
            if answer_entity["id"] in [item["id"] for item in ev["wikidata_entities"]]
        ]
        answering_evidences = sorted(answering_evidences, key=lambda j: j["score"], reverse=True)

        # pad evidences to same number
        evidences_captured = set([evidence["evidence_text"] for evidence in answering_evidences])
        if len(answering_evidences) < num_explaining_evidences:
            additional_evidences = sorted(top_evidences, key=lambda j: j["score"], reverse=True)
            for ev in additional_evidences:
                if len(answering_evidences) == num_explaining_evidences:
                    break
                if not ev["evidence_text"] in evidences_captured:
                    answering_evidences.append(ev)
                    evidences_captured.add(ev["evidence_text"])

        return answering_evidences

    def load(self):
        """Load models."""
        if not self.model_loaded:
            # initialize and load GNNs
            self.configs = [
                self._prepare_config_for_iteration(config_at_i)
                for config_at_i in self.config["gnn_inference"]
            ]
            self.gnns = [
                GNNModule(self.configs[i], iteration=i + 1)
                for i in range(len(self.config["gnn_inference"]))
            ]
            for gnn in self.gnns:
                gnn.load()

            # remember that model is loaded
            self.model_loaded = True


def main():
    if len(sys.argv) < 2:
        raise Exception(
            "python faith/heterogeneous_answering/graph_neural_network/iterative_gnns.py --<FUNCTION> <PATH_TO_CONFIG> [<SOURCES_STR>]"
        )

    function = sys.argv[1]
    config_path = sys.argv[2]
    sources_str = sys.argv[3] if len(sys.argv) > 3 else "kb_text_table_info"
    config = get_config(config_path)

    if function == "--train":
        # train
        gnn = IterativeGNNs(config)
        sources = sources_str.split("_")
        gnn.train(sources=sources)

    elif function == "--test":
        gnn = IterativeGNNs(config)
        sources = sources_str.split("_")
        gnn.test(sources=sources)

    elif function == "--inference":
        gnn = IterativeGNNs(config)
        sources = sources_str.split("_")
        gnn.inference(sources=sources)

    elif function == "--dev":
        gnn = IterativeGNNs(config)
        sources = sources_str.split("_")
        gnn.dev(sources=sources)


if __name__ == "__main__":
    main()
