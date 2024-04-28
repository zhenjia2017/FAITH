import os
import sys
import json

from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance

from faith.library.utils import store_json_with_mkdir, store_jsonl_with_mkdir, get_logger, get_result_logger, get_config

class HeterogeneousAnswering:
    def __init__(self, config):
        """Initialize HA module."""
        self.config = config
        self.logger = get_logger(__name__, config)
        self.result_logger = get_result_logger(config)

    def train(self, sources=["kb", "text", "table", "info"]):
        """Method used in case no training required for HA phase."""
        self.logger.info("Module used does not require training.")

    def inference(self, sources=["kb", "text", "table", "info"]):
        """Run HA on data and add answers for each source combination."""
        tqu = self.config["tqu"]
        fer = self.config["fer"]
        ha = self.config["ha"]
        sources_str = "_".join(sources)

        method_name = self.config["name"]
        evs_max_evidences = self.config["evs_max_evidences"]

        if ha == "seq2seq_ha":
            gnn_max_evidences = ''
        else:
            gnn_max_evidences = []
            for i in range(len(self.config["gnn_inference"])):
                gnn_max_evidences.append(str(self.config["gnn_inference"][i]["gnn_max_evidences"]))

            gnn_max_evidences = '_'.join(gnn_max_evidences)
            print(gnn_max_evidences)

        input_dir = self.config["path_to_intermediate_results"]
        input_path = os.path.join(
            input_dir,
            tqu,
            fer,
            sources_str,
            f"test_ers-{method_name}-{evs_max_evidences}.jsonl"
        )

        output_path = os.path.join(
            input_dir,
            tqu,
            fer,
            sources_str,
            f"test_ers-{method_name}-{evs_max_evidences}-{ha}-{gnn_max_evidences}.json"
        )
        self.inference_on_data_split(input_path, output_path, sources)
        self.evaluate_hs_results(output_path)
        # input_dir = self.config["path_to_intermediate_results"]
        # input_path = os.path.join(
        #     input_dir,
        #     tqu,
        #     fer,
        #     sources_str,
        #     ann_method,
        #     f"p{clocq_p}k{clocq_k}",
        #     f"test_ers-{method_name}-{evs_max_evidences}.jsonl"
        # )
        #
        # output_path = os.path.join(
        #     input_dir,
        #     tqu,
        #     fer,
        #     sources_str,
        #     ann_method,
        #     f"p{clocq_p}k{clocq_k}",
        #     f"test_ers-{method_name}-{evs_max_evidences}-{ha}-{gnn_max_evidences}.json"
        # )
        # self.inference_on_data_split(input_path, output_path, sources)
        # self.evaluate_hs_results(output_path)
        #
        input_path = os.path.join(
            input_dir,
            tqu,
            fer,
            sources_str,
            f"dev_ers-{method_name}-{evs_max_evidences}.jsonl"
        )

        output_path = os.path.join(
            input_dir,
            tqu,
            fer,
            sources_str,
            f"dev_ers-{method_name}-{evs_max_evidences}-{ha}-{gnn_max_evidences}.json"
        )
        #
        self.inference_on_data_split(input_path, output_path, sources)
        self.evaluate_hs_results(output_path)

    def inference_on_data_split(self, input_path, output_path, sources, jsonl=False):
        """Run HA on given data split."""
        # open data
        input_data = list()
        with open(input_path, "r") as fp:
            line = fp.readline()
            while line:
                instance = json.loads(line)
                input_data.append(instance)
                line = fp.readline()

        # inference
        self.inference_on_data(input_data, sources)

        # store processed data
        if jsonl:
            store_jsonl_with_mkdir(input_data, output_path)
        else:
            store_json_with_mkdir(input_data, output_path)

    def inference_on_data(self, input_data, sources):
        """Run HA on the given data."""
        for instance in tqdm(input_data):
            self.inference_on_instance(instance)

        self._log_results(input_data, sources)
       

    def inference_on_instance(self, instance):
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )

    def get_ranked_answers(self, generated_answer, instance):
        """
        Convert the generated answer text to a list of Wikidata IDs,
        and return the ranked answers.
        Can be used for any method that predicts an answer string (instead of a KB item).
        """
        # check if existential (special treatment)
        # question = instance["Question"]
        if generated_answer is None:
            return [{"answer": {"id": "None", "label": "None"}, "rank": 1, "score": 0.0}]
        smallest_diff = 100000
        all_answers = list()
        mentions = set()
        for evidence in instance["candidate_evidences"]:
            for disambiguation in evidence["disambiguations"]:
                mention = disambiguation[0]
                id = disambiguation[1]
                if id is None or id == False:
                    continue

                # skip duplicates
                ans = str(mention) + str(id)
                if ans in mentions:
                    continue
                mentions.add(ans)
                # exact match
                if generated_answer == mention:
                    diff = 0
                # otherwise compute edit distance
                else:
                    diff = levenshtein_distance(generated_answer, mention)

                all_answers.append({"answer": {"id": id, "label": mention}, "score": diff})

        sorted_answers = sorted(all_answers, key=lambda j: j["score"])
        ranked_answers = [
            {"answer": answer["answer"], "score": answer["score"], "rank": i + 1}
            for i, answer in enumerate(sorted_answers)
        ]

        # don't return all answers
        max_answers = self.config["ha_max_answers"]
        ranked_answers = ranked_answers[:max_answers]
        if not ranked_answers:
            ranked_answers = [{"answer": {"id": "None", "label": "None"}, "score": 0.0, "rank": 1}]
        return ranked_answers

    def evaluate_hs_results_remove_ordinal(self, results_path, sources_str):
        """
        Evaluate the results of the retrieval phase, for
        each source, and for each category.
        """
        #sources_str = "kb_text_table_info"
        category_to_p1 = {"explicit": 0, "implicit": 0, "temp.ans": 0, "all": 0}
        category_to_h5 = {"explicit": 0, "implicit": 0, "temp.ans": 0, "all": 0}
        category_to_mrr = {"explicit": 0, "implicit": 0, "temp.ans": 0, "all": 0}

        instance_remove_ordinal = []
        # process data
        with open(results_path, "r") as fp:
            input_data = json.load(fp)

            for instance in input_data:
                if "Ordinal" in instance["Temporal question type"]: continue
                if type(instance["Temporal question type"]) != list:
                    instance["Temporal question type"] = [instance["Temporal question type"]]
                instance_remove_ordinal.append(instance)

        # compute results
        for category in category_to_p1:
            if category == "all":
                # drop ordinal type
                p_at_1_list = [
                    instance["p_at_1"]
                    for instance in instance_remove_ordinal
                ]
                # p_at_1_list = [instance["p_at_1"] for instance in input_data]
            else:

                p_at_1_list = [
                    instance["p_at_1"]
                    for instance in instance_remove_ordinal
                    if category in [item.lower() for item in instance["Temporal question type"]]
                ]

            if len(p_at_1_list) == 0: continue
            p_at_1 = sum(p_at_1_list) / len(p_at_1_list)
            p_at_1 = round(p_at_1, 3)
            num_questions = len(p_at_1_list)
            # log result
            category_to_p1[category] = p_at_1
            res_str = f"Gold answers - {category} - {sources_str} - P@1 ({num_questions}): {p_at_1}"
            self.logger.info(res_str)

        for category in category_to_mrr:
            # compute results
            if category == "all":
                mrr_list = [
                    instance["mrr"]
                    for instance in instance_remove_ordinal
                ]
            else:
                mrr_list = [
                    instance["mrr"]
                    for instance in instance_remove_ordinal
                    if category in [item.lower() for item in instance["Temporal question type"]]
                ]

            if len(mrr_list) == 0: continue
            mrr = sum(mrr_list) / len(mrr_list)
            mrr = round(mrr, 3)
            category_to_mrr[category] = mrr
            # log result
            res_str = f"Gold answers - {category} - {sources_str} - MRR ({num_questions}): {mrr}"
            self.logger.info(res_str)

        for category in category_to_h5:
            if category == "all":
                # compute results
                hit_at_5_list = [
                    instance["h_at_5"]
                    for instance in instance_remove_ordinal

                ]
            else:
                hit_at_5_list = [
                    instance["h_at_5"]
                    for instance in instance_remove_ordinal
                    if category in [item.lower() for item in instance["Temporal question type"]]
                ]

            if len(hit_at_5_list) == 0: continue
            hit_at_5 = sum(hit_at_5_list) / len(hit_at_5_list)
            hit_at_5 = round(hit_at_5, 3)
            category_to_h5[category] = hit_at_5
            # log result
            res_str = f"Gold answers - {category} - {sources_str} - H@5 ({num_questions}): {hit_at_5}"
            self.logger.info(res_str)

        # print results
        res_path = results_path.replace(".json", "_remove_ordinal.res")

        with open(res_path, "w") as fp:
            fp.write(f"ha evaluation result:\n")
            for category in category_to_p1:
                fp.write("\n")
                fp.write(f"category: {category}\n")
                fp.write(f"P@1: {category_to_p1[category]}\n")
            for category in category_to_h5:
                fp.write("\n")
                fp.write(f"category: {category}\n")
                fp.write(f"H@5: {category_to_h5[category]}\n")
            for category in category_to_mrr:
                fp.write("\n")
                fp.write(f"category: {category}\n")
                fp.write(f"MRR: {category_to_mrr[category]}\n")

    def _get_data_dirs(self, sources):
        data_dir = self.config["path_to_intermediate_results"]
        tqu = self.config["tqu"]
        fer = self.config["fer"]
        ha = self.config["ha"]
        sources_str = "_".join(sources)
        input_dir, output_dir = os.path.join(data_dir, tqu, fer, sources_str, ha)
        return input_dir, output_dir

    def evaluate_implicit_hs_results(self, results_path):
        """
        Evaluate the results of the retrieval phase, for
        each source, and for each category.
        """
        sources_str = "kb_text_table_info"

        category_to_p1 = {"explicit": 0, "implicit": 0, "ordinal": 0, "temporal value": 0, "all": 0}
        category_to_h5 = {"explicit": 0, "implicit": 0, "ordinal": 0, "temporal value": 0, "all": 0}
        category_to_mrr = {
            "explicit": 0,
            "implicit": 0,
            "ordinal": 0,
            "temporal value": 0,
            "all": 0,
        }

        # process data
        with open(results_path, "r") as fp:
            input_data = json.load(fp)

        # compute results
        for category in category_to_p1:
            if category == "all":
                # drop ordinal type
                p_at_1_list = [
                    instance["p_at_1"]
                    for instance in input_data
                    if "ordinal"
                    not in [
                        item.lower() for item in instance["structured_temporal_form"]["category"]
                    ]
                ]
            else:
                p_at_1_list = [
                    instance["p_at_1"]
                    for instance in input_data
                    if category
                    in [item.lower() for item in instance["structured_temporal_form"]["category"]]
                ]
            p_at_1 = sum(p_at_1_list) / len(p_at_1_list)
            p_at_1 = round(p_at_1, 3)
            num_questions = len(p_at_1_list)
            # log result
            category_to_p1[category] = p_at_1
            res_str = f"Gold answers - {category} - {sources_str} - P@1 ({num_questions}): {p_at_1}"
            self.logger.info(res_str)

        for category in category_to_mrr:
            # compute results
            if category == "all":
                mrr_list = [
                    instance["mrr"]
                    for instance in input_data
                    if "ordinal"
                    not in [
                        item.lower() for item in instance["structured_temporal_form"]["category"]
                    ]
                ]
            else:
                mrr_list = [
                    instance["mrr"]
                    for instance in input_data
                    if category
                    in [item.lower() for item in instance["structured_temporal_form"]["category"]]
                ]
            mrr = sum(mrr_list) / len(mrr_list)
            mrr = round(mrr, 3)
            category_to_mrr[category] = mrr
            # log result
            res_str = f"Gold answers - {category} - {sources_str} - MRR ({num_questions}): {mrr}"
            self.logger.info(res_str)

        for category in category_to_h5:
            if category == "all":
                # compute results
                hit_at_5_list = [
                    instance["h_at_5"]
                    for instance in input_data
                    if "Ordinal"
                    not in [
                        item.lower() for item in instance["structured_temporal_form"]["category"]
                    ]
                ]
            else:
                hit_at_5_list = [
                    instance["h_at_5"]
                    for instance in input_data
                    if category
                    in [item.lower() for item in instance["structured_temporal_form"]["category"]]
                ]
            hit_at_5 = sum(hit_at_5_list) / len(hit_at_5_list)
            hit_at_5 = round(hit_at_5, 3)
            category_to_h5[category] = hit_at_5
            # log result
            res_str = f"Gold answers - {sources_str} - H@5 ({num_questions}): {hit_at_5}"
            self.logger.info(res_str)

        # print results
        res_path = results_path.replace(".json", "-ha-drop-ordinal-predict-implicit.res")

        with open(res_path, "w") as fp:
            fp.write(f"ha evaluation result:\n")
            for category in category_to_p1:
                fp.write("\n")
                fp.write(f"category: {category}\n")
                fp.write(f"P@1: {category_to_p1[category]}\n")
            for category in category_to_h5:
                fp.write("\n")
                fp.write(f"category: {category}\n")
                fp.write(f"H@5: {category_to_h5[category]}\n")
            for category in category_to_mrr:
                fp.write("\n")
                fp.write(f"category: {category}\n")
                fp.write(f"MRR: {category_to_mrr[category]}\n")


    def evaluate_hs_results(self, results_path, sources_str):
        """
        Evaluate the results of the retrieval phase, for
        each source, and for each category.
        """
        #sources_str = "kb_text_table_info"
        category_to_p1 = {"ordinal": 0, "explicit": 0, "implicit": 0, "temp.ans": 0, "all": 0}
        category_to_h5 = {"ordinal": 0, "explicit": 0, "implicit": 0, "temp.ans": 0, "all": 0}
        category_to_mrr = {"ordinal": 0, "explicit": 0, "implicit": 0, "temp.ans": 0, "all": 0}

        instance_remove_ordinal = []
        # process data
        with open(results_path, "r") as fp:
            input_data = json.load(fp)

            for instance in input_data:
                # if "Ordinal" in instance["Temporal question type"]: continue
                if type(instance["Temporal question type"]) != list:
                    instance["Temporal question type"] = [instance["Temporal question type"]]
                instance_remove_ordinal.append(instance)

        # compute results
        for category in category_to_p1:
            if category == "all":
                # drop ordinal type
                p_at_1_list = [
                    instance["p_at_1"]
                    for instance in instance_remove_ordinal
                ]
                # p_at_1_list = [instance["p_at_1"] for instance in input_data]
            else:

                p_at_1_list = [
                    instance["p_at_1"]
                    for instance in instance_remove_ordinal
                    if category in [item.lower() for item in instance["Temporal question type"]]
                ]

            if len(p_at_1_list) == 0: continue
            p_at_1 = sum(p_at_1_list) / len(p_at_1_list)
            p_at_1 = round(p_at_1, 3)
            num_questions = len(p_at_1_list)
            # log result
            category_to_p1[category] = p_at_1
            res_str = f"Gold answers - {category} - {sources_str} - P@1 ({num_questions}): {p_at_1}"
            self.logger.info(res_str)

        for category in category_to_mrr:
            # compute results
            if category == "all":
                mrr_list = [
                    instance["mrr"]
                    for instance in instance_remove_ordinal
                ]
            else:
                mrr_list = [
                    instance["mrr"]
                    for instance in instance_remove_ordinal
                    if category in [item.lower() for item in instance["Temporal question type"]]
                ]

            if len(mrr_list) == 0: continue
            mrr = sum(mrr_list) / len(mrr_list)
            mrr = round(mrr, 3)
            category_to_mrr[category] = mrr
            # log result
            res_str = f"Gold answers - {category} - {sources_str} - MRR ({num_questions}): {mrr}"
            self.logger.info(res_str)

        for category in category_to_h5:
            if category == "all":
                # compute results
                hit_at_5_list = [
                    instance["h_at_5"]
                    for instance in instance_remove_ordinal

                ]
            else:
                hit_at_5_list = [
                    instance["h_at_5"]
                    for instance in instance_remove_ordinal
                    if category in [item.lower() for item in instance["Temporal question type"]]
                ]

            if len(hit_at_5_list) == 0: continue
            hit_at_5 = sum(hit_at_5_list) / len(hit_at_5_list)
            hit_at_5 = round(hit_at_5, 3)
            category_to_h5[category] = hit_at_5
            # log result
            res_str = f"Gold answers - {category} - {sources_str} - H@5 ({num_questions}): {hit_at_5}"
            self.logger.info(res_str)

        # print results
        res_path = results_path.replace(".json", ".res")

        with open(res_path, "w") as fp:
            fp.write(f"ha evaluation result:\n")
            for category in category_to_p1:
                fp.write("\n")
                fp.write(f"category: {category}\n")
                fp.write(f"P@1: {category_to_p1[category]}\n")
            for category in category_to_h5:
                fp.write("\n")
                fp.write(f"category: {category}\n")
                fp.write(f"H@5: {category_to_h5[category]}\n")
            for category in category_to_mrr:
                fp.write("\n")
                fp.write(f"category: {category}\n")
                fp.write(f"MRR: {category_to_mrr[category]}\n")


    def _log_results(self, input_data, sources):
        # compute results
        sources_str = "_".join(sources)
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





