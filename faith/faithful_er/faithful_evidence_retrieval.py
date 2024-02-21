import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from faith.library.utils import get_config, get_logger
from faith.library.string_library import StringLibrary
from faith.evaluation import answer_presence


class FaithfulEvidenceRetrieval:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, self.config)
        self.tsf_delimiter = self.config["tsf_delimiter"]
        self.string_lib = StringLibrary(config)
        self.faith_or_unfaith = self.config["faith_or_unfaith"]

    def train(self, sources=None):
        """ Abstract training function that triggers training of submodules. """
        self.logger.info("Module used does not require training.")

    def er_inference(self, sources=["kb", "text", "table", "info"]):
        """Run ER on data and add retrieve top-e evidences for each source combination."""
        benchmark = self.config["benchmark"]
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]
        tqu = self.config["tqu"]
        method_name = self.config["name"]
        fer = self.config["fer"]
        sources_str = "_".join(sources)

        start = time.time()
        input_path = os.path.join(input_dir, benchmark, tqu, f"dev_tqu-{method_name}.json")
        output_path = os.path.join(output_dir, benchmark, tqu, fer, sources_str, f"dev_er-{method_name}.jsonl")
        self.er_inference_on_data_split(input_path, output_path, sources)
        self.evaluate_retrieval_results(output_path)
        running_time = time.time() - start
        self.logger.info(f"retrieval time for the dev set is {running_time}")

        start = time.time()
        input_path = os.path.join(input_dir, benchmark, tqu, f"train_tqu-{method_name}.json")
        output_path = os.path.join(output_dir, benchmark, tqu, fer, sources_str, f"train_er-{method_name}.jsonl")
        self.er_inference_on_data_split(input_path, output_path, sources)
        self.evaluate_retrieval_results(output_path)
        running_time = time.time() - start
        self.logger.info(f"retrieval time for the train set is {running_time}")

    def er_inference_on_data_split(self, input_path, output_path, sources):
        """
        Run ERP on the dataset to predict
        answering evidences for each TSF in the dataset.
        """
        # open data
        with open(input_path, "r") as fp:
            data = json.load(fp)
        self.logger.info(f"Input data loaded from: {input_path}.")
        self.logger.info(f"Length of input data: {len(data)}.")
        # create folder if not exists
        output_dir = os.path.dirname(output_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        retrieved_ids = []
        if os.path.exists(output_path):
            with open(output_path, "r") as fp:

                for line in tqdm(fp):
                    try:
                        instance = json.loads(line)
                    except:
                        print("Load json error!")
                        continue
                    retrieved_ids.append(instance["Id"])

                print(len(retrieved_ids))
                print(retrieved_ids[-1])

        with open(output_path, "a") as fp:
            for instance in tqdm(data):
                ques_id = instance["Id"]
                if ques_id in retrieved_ids:
                    print(f"{ques_id}already retrieved")
                    continue
                print(f"Start retrieve {ques_id}")
                evidences = self.er_inference_on_instance(instance, sources)
                # answer presence
                if "answers" not in instance:
                    instance["answers"] = self.string_lib.format_answers(instance)
                hit, answering_evidences = answer_presence(evidences, instance["answers"])
                instance["answer_presence"] = hit
                instance["answer_presence_per_src"] = {
                    evidence["source"]: 1 for evidence in answering_evidences
                }

                # write instance to file
                fp.write(json.dumps(instance))
                fp.write("\n")

        # log
        self.logger.info(f"Evaluating retrieval results: {output_path}.")
        self.evaluate_retrieval_results(output_path, sources)
        self.logger.info(f"Done with processing: {input_path}.")

    def prune_on_data_split(self, input_path, output_path, sources):
        with open(input_path, 'r') as fin, open(output_path, "w") as fp:
            for line in tqdm(fin):
                try:
                    instance = json.loads(line)
                except:
                    print("Load json error!")
                    continue
                faithful_evidences = self.prune_on_instance(instance, sources)
                instance["candidate_evidences"] = faithful_evidences
                hit, answering_evidences = answer_presence(faithful_evidences, instance["answers"])
                instance["answer_presence"] = hit
                instance["answer_presence_per_src"] = {
                    evidence["source"]: 1 for evidence in answering_evidences
                }

                # write instance to file
                fp.write(json.dumps(instance))
                fp.write("\n")

            # log
        self.logger.info(f"Evaluating retrieval results: {output_path}.")
        self.evaluate_retrieval_results(output_path, sources)
        self.logger.info(f"Done with processing: {input_path}.")

    def prune_on_instance(self, input_data, sources=["kb", "text", "table", "info"]):
        """Retrieve candidate and prune for generating faithful evidences for TSF."""
        raise Exception("This is an abstract function which should be overwritten in a derived class!")

    def inference_on_data(self, input_data, sources=["kb", "text", "table", "info"]):
        """Run model on data and add predictions."""
        # model inference on given data
        for instance in tqdm(input_data):
            self.inference_on_instance(instance, sources)
        return input_data

    def inference_on_instance(self, instance, sources=["kb", "text", "table", "info"]):
        """Retrieve candidate and prune for generating faithful evidences for TSF."""
        raise Exception("This is an abstract function which should be overwritten in a derived class!")

    def er_inference_on_instance(self, instance, sources=["kb", "text", "table", "info"]):
        """Retrieve candidate and prune for generating faithful evidences for TSF."""
        raise Exception("This is an abstract function which should be overwritten in a derived class!")

    def evs_inference_on_instance(self, instance, max_evidence=100, sources=["kb", "text", "table", "info"]):
        """Run model on data and add predictions."""
        # inference: add predictions to data
        """ Abstract evidence scoring function that triggers es_inference_on_instance of submodules. """
        raise Exception("This is an abstract function which should be overwritten in a derived class!")

    def evs_inference(self, sources=["kb", "text", "table", "info"]):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]
        benchmark = self.config["benchmark"]
        faith = self.config["faith_or_unfaith"]
        tqu = self.config["tqu"]
        fer = self.config["fer"]
        method_name = self.config["name"]
        evs_model = self.config["evs_model"]
        max_evidence = self.config["evs_max_evidences"]
        # either use given option, or from config
        sources_string = "_".join(sources)
        input_dir = os.path.join(input_dir, benchmark, tqu, fer, sources_string)
        output_dir = os.path.join(output_dir, benchmark, tqu, fer, sources_string)

        if faith == "faith":
            input_path = os.path.join(input_dir, f"dev_erp-{method_name}.jsonl")
            output_path = os.path.join(output_dir, f"dev_erps-{method_name}-{evs_model}-{max_evidence}.jsonl")
        else:
            input_path = os.path.join(input_dir, f"dev_er-{method_name}.jsonl")
            output_path = os.path.join(output_dir, f"dev_ers-{method_name}-{evs_model}-{max_evidence}.jsonl")
        self.evs_inference_on_data_split(input_path, output_path, max_evidence=max_evidence)
        self.evaluate_retrieval_results(output_path)

        if faith == "faith":
            input_path = os.path.join(input_dir, f"train_erp-{method_name}.jsonl")
            output_path = os.path.join(output_dir, f"train_erps-{method_name}-{evs_model}-{max_evidence}.jsonl")
        else:
            input_path = os.path.join(input_dir, f"train_er-{method_name}.jsonl")
            output_path = os.path.join(output_dir, f"train_ers-{method_name}-{evs_model}-{max_evidence}.jsonl")
        self.evs_inference_on_data_split(input_path, output_path, max_evidence=max_evidence)
        self.evaluate_retrieval_results(output_path)

    def evs_inference_on_data_split(self, input_path, output_path, max_evidence=100,
                                    sources=["kb", "text", "table", "info"]):
        # score
        # create folder if not exists
        output_dir = os.path.dirname(output_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ans_pres_initial = list()
        ans_pres_sbert = list()
        # process data
        with open(input_path, 'r') as fp:
            # process data
            with open(output_path, "w") as fpout:
                for line in tqdm(fp):
                    try:
                        instance = json.loads(line)
                    except:
                        print("Load json error!")
                        continue
                    ans_pres = instance["answer_presence"]
                    ans_pres_initial.append(ans_pres)
                    top_evidences = self.evs_inference_on_instance(instance, max_evidence=max_evidence, sources=sources)
                    hit, answering_evidences = answer_presence(top_evidences, instance["answers"])
                    instance["answer_presence"] = hit
                    instance["answer_presence_per_src"] = {
                        evidence["source"]: 1 for evidence in answering_evidences
                    }
                    ans_pres_sbert.append(hit)
                    # write instance to file
                    fpout.write(json.dumps(instance))
                    fpout.write("\n")
        evs_model = self.config["evs_model"]
        print(f"Initial ans. pres.: {sum(ans_pres_initial) / len(ans_pres_initial)} ({len(ans_pres_initial)})")
        print(f"Ans. pres. after {evs_model}: {sum(ans_pres_sbert) / len(ans_pres_sbert)} ({len(ans_pres_sbert)})")

    def store_cache(self):
        """Store cache of evidence retriever."""
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )

    def evaluate_retrieval_results_res_stage(self, results_path, stage="initial",
                                             sources=['kb', 'info', 'table', 'text']):
        """
                Evaluate the results of the retrieval phase, for
                each source, and for each category.
                """
        # score
        if stage == "scoring":
            return self.evaluate_retrieval_results_res(results_path, sources=sources)

        if stage == "pruning" and self.faith_or_unfaith == "unfaith":
            return self.evaluate_retrieval_results_res(results_path, sources=sources)

        answer_presences = list()
        source_to_ans_pres = {source: 0 for source in sources}
        source_to_ans_pres.update({"all": 0})
        category_to_ans_pres = {"explicit": [], "ordinal": [], "implicit": [], "temp.ans": [], "all": []}

        total_source_num = {source: [] for source in sources}
        total_source_num.update({"all": []})

        with open(results_path, "r") as fp:
            data = json.load(fp)
            for instance in tqdm(data):
                source_to_evidence_num = {source: 0 for source in sources}
                category_slot = [item.lower() for item in instance["Temporal question type"]]

                for source in sources:
                    total_source_num[source].append(source_to_evidence_num[source])
                    total_source_num["all"].append(source_to_evidence_num[source])

                hit = instance[f"answer_presence_{stage}"]
                answer_presence_per_src = instance[f"answer_presence_per_src_{stage}"]

                category_to_ans_pres["all"] += [hit]

                for category in category_to_ans_pres.keys():
                    if category in category_slot:
                        category_to_ans_pres[category] += [hit]

                answer_presences += [hit]

                for src, ans_presence in answer_presence_per_src.items():
                    source_to_ans_pres[src] += ans_presence
                # aggregate overall answer presence for validation
                if len(answer_presence_per_src.items()):
                    source_to_ans_pres["all"] += 1

        # print results
        res_path = results_path.replace(".jsonl", f"-retrieval_{stage}.res")
        with open(res_path, "w") as fp:
            fp.write(f"evaluation result:\n")
            avg_answer_presence = sum(answer_presences) / len(answer_presences)
            fp.write(f"Avg. answer presence: {avg_answer_presence}\n")
            answer_presence_per_src = {
                src: (num / len(answer_presences)) for src, num in source_to_ans_pres.items()
            }
            fp.write(f"Answer presence per source: {answer_presence_per_src}")

            fp.write("\n")
            fp.write("\n")
            category_answer_presence_per_src = {
                category: (sum(num) / len(num)) for category, num in category_to_ans_pres.items() if len(num) != 0
            }
            fp.write(f"Category Answer presence per source: {category_answer_presence_per_src}")

    def evaluate_retrieval_results(self, results_path, sources=['kb', 'info', 'table', 'text']):
        """
        Evaluate the results of the retrieval phase, for
        each source, and for each category.
        """
        # score
        answer_presences = list()
        source_to_ans_pres = {source: 0 for source in sources}
        source_to_ans_pres.update({"all": 0})
        category_to_ans_pres = {"ordinal": [], "explicit": [], "implicit": [], "temp.ans": [], "all": []}
        category_to_evi_num = {"ordinal": [], "explicit": [], "implicit": [], "temp.ans": [], "all": []}

        total_source_num = {source: [] for source in sources}
        total_source_num.update({"all": []})

        # process data
        data_num = 0
        with open(results_path, 'r') as fp:
            for line in tqdm(fp):
                try:
                    instance = json.loads(line)
                    data_num += 1
                except:
                    print("error when loads line!")
                    continue
                candidate_evidences = instance["candidate_evidences"]
                source_to_evidence_num = {source: 0 for source in sources}
                category_slot = [item.lower() for item in instance["Temporal question type"]]
                for evidence in candidate_evidences:
                    source_to_evidence_num[evidence["source"]] += 1

                for source in sources:
                    total_source_num[source].append(source_to_evidence_num[source])
                    total_source_num["all"].append(source_to_evidence_num[source])

                hit, answering_evidences = answer_presence(candidate_evidences, instance["answers"])

                answer_presence_per_src = {
                    evidence["source"]: 1 for evidence in answering_evidences
                }

                category_to_ans_pres["all"] += [hit]
                category_to_evi_num["all"] += [len(candidate_evidences)]
                for category in category_to_ans_pres.keys():
                    if category in category_slot:
                        category_to_evi_num[category] += [len(candidate_evidences)]
                        category_to_ans_pres[category] += [hit]

                answer_presences += [hit]

                for src, ans_presence in answer_presence_per_src.items():
                    source_to_ans_pres[src] += ans_presence
                # aggregate overall answer presence for validation
                if len(answer_presence_per_src.items()):
                    source_to_ans_pres["all"] += 1

        # save results
        res_path = results_path.replace(".jsonl", ".res")
        with open(res_path, "w") as fp:

            fp.write(f"evaluation result: for instances of {data_num}\n")
            for source in total_source_num:
                fp.write(f"source: {source}\n")
                fp.write(f"Avg. evidence number: {sum(total_source_num[source]) / len(total_source_num[source])}\n")
                sorted_source_num = total_source_num[source]
                sorted_source_num.sort()
                fp.write(f"Max. evidence number: {sorted_source_num[-1]}\n")
                fp.write(f"Min. evidence number: {sorted_source_num[0]}\n")

            avg_answer_presence = sum(answer_presences) / len(answer_presences)
            fp.write(f"Avg. answer presence: {avg_answer_presence}\n")
            answer_presence_per_src = {
                src: (num / len(answer_presences)) for src, num in source_to_ans_pres.items()
            }
            fp.write(f"Answer presence per source: {answer_presence_per_src}")

            fp.write("\n")
            fp.write("\n")
            category_answer_presence_per_src = {
                category: (sum(num) / len(num)) for category, num in category_to_ans_pres.items() if len(num) != 0
            }
            fp.write(f"Category Answer presence per source: {category_answer_presence_per_src}")

            for category in category_to_evi_num:
                fp.write("\n")
                try:
                    fp.write(f"category: {category}\n")
                    fp.write(
                        f"Avg. evidence number: {sum(category_to_evi_num[category]) / len(category_to_evi_num[category])}\n")
                    sorted_category_num = category_to_evi_num[category]
                    sorted_category_num.sort()
                    fp.write(f"Max. evidence number: {sorted_category_num[-1]}\n")
                    fp.write(f"Min. evidence number: {sorted_category_num[0]}\n")
                except:
                    fp.write(f"category: {category} not in the corpus\n")
                    continue

    def evaluate_retrieval_results_res(self, results_path, sources=['kb', 'info', 'table', 'text']):
        """
                Evaluate the results of the retrieval phase, for
                each source, and for each category.
                """
        # score
        answer_presences = list()
        source_to_ans_pres = {source: 0 for source in sources}
        source_to_ans_pres.update({"all": 0})
        category_to_ans_pres = {"explicit": [], "ordinal": [], "implicit": [], "temp.ans": [], "all": []}
        category_to_evi_num = {"explicit": [], "ordinal": [], "implicit": [], "temp.ans": [], "all": []}

        total_source_num = {source: [] for source in sources}
        total_source_num.update({"all": []})

        with open(results_path, "r") as fp:
            data = json.load(fp)
            for instance in tqdm(data):
                candidate_evidences = instance["candidate_evidences"]
                source_to_evidence_num = {source: 0 for source in sources}
                if type(instance["Temporal question type"]) != list:
                    instance["Temporal question type"] = [instance["Temporal question type"]]
                category_slot = [item.lower() for item in instance["Temporal question type"]]
                # if "ordinal" in category_slot: continue
                for evidence in candidate_evidences:
                    source_to_evidence_num[evidence["source"]] += 1

                for source in sources:
                    total_source_num[source].append(source_to_evidence_num[source])
                    total_source_num["all"].append(source_to_evidence_num[source])

                hit, answering_evidences = answer_presence(candidate_evidences, instance["answers"])

                answer_presence_per_src = {
                    evidence["source"]: 1 for evidence in answering_evidences
                }

                category_to_ans_pres["all"] += [hit]
                category_to_evi_num["all"] += [len(candidate_evidences)]
                for category in category_to_ans_pres.keys():
                    if category in category_slot:
                        category_to_evi_num[category] += [len(candidate_evidences)]
                        category_to_ans_pres[category] += [hit]

                answer_presences += [hit]

                for src, ans_presence in answer_presence_per_src.items():
                    source_to_ans_pres[src] += ans_presence
                # aggregate overall answer presence for validation
                if len(answer_presence_per_src.items()):
                    source_to_ans_pres["all"] += 1

        # print results
        res_path = results_path.replace(".jsonl", "-retrieval.res")
        with open(res_path, "w") as fp:
            fp.write(f"evaluation result:\n")
            for source in total_source_num:
                fp.write(f"source: {source}\n")
                fp.write(f"Avg. evidence number: {sum(total_source_num[source]) / len(total_source_num[source])}\n")
                sorted_source_num = total_source_num[source]
                sorted_source_num.sort()
                fp.write(f"Max. evidence number: {sorted_source_num[-1]}\n")
                fp.write(f"Min. evidence number: {sorted_source_num[0]}\n")

            avg_answer_presence = sum(answer_presences) / len(answer_presences)
            fp.write(f"Avg. answer presence: {avg_answer_presence}\n")
            answer_presence_per_src = {
                src: (num / len(answer_presences)) for src, num in source_to_ans_pres.items()
            }
            fp.write(f"Answer presence per source: {answer_presence_per_src}")

            fp.write("\n")
            fp.write("\n")
            category_answer_presence_per_src = {
                category: (sum(num) / len(num)) for category, num in category_to_ans_pres.items() if len(num) != 0
            }
            fp.write(f"Category Answer presence per source: {category_answer_presence_per_src}")

            for category in category_to_evi_num:
                fp.write("\n")
                try:
                    fp.write(f"category: {category}\n")
                    fp.write(
                        f"Avg. evidence number: {sum(category_to_evi_num[category]) / len(category_to_evi_num[category])}\n")
                    sorted_category_num = category_to_evi_num[category]
                    sorted_category_num.sort()
                    fp.write(f"Max. evidence number: {sorted_category_num[-1]}\n")
                    fp.write(f"Min. evidence number: {sorted_category_num[0]}\n")
                except:
                    fp.write(f"category: {category} not in the corpus\n")
                    continue

    def evaluate_scoring_results(self, results_path, sources=['kb', 'info', 'table', 'text']):
        """
        Evaluate the results of the retrieval phase, for
        each source, and for each category.
        """
        # score
        answer_presences = list()
        source_to_ans_pres = {source: 0 for source in sources}
        source_to_ans_pres.update({"all": 0})
        category_to_ans_pres = {"explicit": [], "implicit": [], "temp.ans": [], "all": []}
        category_to_evi_num = {"explicit": [], "implicit": [], "temp.ans": [], "all": []}

        total_source_num = {source: [] for source in sources}
        total_source_num.update({"all": []})

        # process data
        with open(results_path, "r") as fp:
            data = json.load(fp)
            for instance in tqdm(data):
                candidate_evidences = instance["candidate_evidences"]
                source_to_evidence_num = {source: 0 for source in sources}
                category_slot = [item.lower() for item in instance["Temporal question type"]]
                if "ordinal" in category_slot: continue
                for evidence in candidate_evidences:
                    source_to_evidence_num[evidence["source"]] += 1

                for source in sources:
                    total_source_num[source].append(source_to_evidence_num[source])
                    total_source_num["all"].append(source_to_evidence_num[source])

                hit, answering_evidences = answer_presence(candidate_evidences, instance["answers"])

                answer_presence_per_src = {
                    evidence["source"]: 1 for evidence in answering_evidences
                }

                category_to_ans_pres["all"] += [hit]
                category_to_evi_num["all"] += [len(candidate_evidences)]
                for category in category_to_ans_pres.keys():
                    if category in category_slot:
                        category_to_evi_num[category] += [len(candidate_evidences)]
                        category_to_ans_pres[category] += [hit]

                answer_presences += [hit]

                for src, ans_presence in answer_presence_per_src.items():
                    source_to_ans_pres[src] += ans_presence
                # aggregate overall answer presence for validation
                if len(answer_presence_per_src.items()):
                    source_to_ans_pres["all"] += 1

        # print results
        res_path = results_path.replace(".jsonl", "-scoring-remove-ordinal.res")
        with open(res_path, "w") as fp:
            fp.write(f"evaluation result:\n")
            for source in total_source_num:
                fp.write(f"source: {source}\n")
                if len(total_source_num[source]) > 0:
                    fp.write(f"Avg. evidence number: {sum(total_source_num[source]) / len(total_source_num[source])}\n")
                    sorted_source_num = total_source_num[source]
                    sorted_source_num.sort()
                    fp.write(f"Max. evidence number: {sorted_source_num[-1]}\n")
                    fp.write(f"Min. evidence number: {sorted_source_num[0]}\n")

            avg_answer_presence = sum(answer_presences) / len(answer_presences)
            fp.write(f"Avg. answer presence: {avg_answer_presence}\n")
            answer_presence_per_src = {
                src: (num / len(answer_presences)) for src, num in source_to_ans_pres.items()
            }
            fp.write(f"Answer presence per source: {answer_presence_per_src}")

            fp.write("\n")
            fp.write("\n")
            category_answer_presence_per_src = {
                category: (sum(num) / len(num)) for category, num in category_to_ans_pres.items() if len(num) != 0
            }
            fp.write(f"Category Answer presence per source: {category_answer_presence_per_src}")

            for category in category_to_evi_num:
                fp.write(f"category: {category}\n")
                fp.write(
                    f"Avg. evidence number: {sum(category_to_evi_num[category]) / len(category_to_evi_num[category])}\n")
                sorted_category_num = category_to_evi_num[category]
                sorted_category_num.sort()
                fp.write(f"Max. evidence number: {sorted_category_num[-1]}\n")
                fp.write(f"Min. evidence number: {sorted_category_num[0]}\n")


