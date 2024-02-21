import os
import json
from pathlib import Path
from tqdm import tqdm


from faith.library.utils import store_json_with_mkdir, get_logger

class TemporalQuestionUnderstanding:
    """Abstract class for TQU phase."""

    def __init__(self, config):
        """Initialize TQU module."""
        self.config = config
        self.logger = get_logger(__name__, config)

    def train(self):
        """Method used in case no training required for TQU phase."""
        self.logger.info("TQU - Module used does not require training.")

    def inference(self, topk_answers=1):
        """Run model on data and add predictions."""
        # inference: add predictions to data
        self.inference_on_data_split("dev", topk_answers)
        self.inference_on_data_split("train", topk_answers)
        self.inference_on_data_split("test", topk_answers)

    def evaluate_tqu(self, file_path_of_dataset_with_generated_tsf):
        signal_to_ac = {"after": [], "before": [], "overlap": [], "finish": [], "start": [], "ordinal": []}
        type_to_ac = {"implicit": [], "non-implicit": []}
        with open(file_path_of_dataset_with_generated_tsf, "r") as fp:
            data = json.load(fp)
            for instance in tqdm(data):

                if type(instance["Temporal question type"]) != list:
                    instance["Temporal question type"] = [instance["Temporal question type"]]
                if type(instance["Temporal signal"]) != list:
                    instance["Temporal signal"] = [instance["Temporal signal"]]

                gold_category = [item.lower() for item in instance["Temporal question type"]]
                gold_signal = [item.lower() for item in instance["Temporal signal"]]
                tsf = instance["structured_temporal_form"]
                predict_category = tsf["category"].lower()
                predict_signal = tsf["temporal_signal"].lower()

                for key in signal_to_ac:
                    if key in gold_signal:
                        if predict_signal in gold_signal:
                            signal_to_ac[key] += [1]
                        else:
                            signal_to_ac[key] += [0]

                if "implicit" in gold_category:
                    if predict_category == "implicit":
                        type_to_ac["implicit"] += [1]
                    else:
                        type_to_ac["implicit"] += [0]

                else:
                    if predict_category == "none":
                        type_to_ac["non-implicit"] += [1]
                    else:
                        type_to_ac["non-implicit"] += [0]


        # open data
        output_path = file_path_of_dataset_with_generated_tsf.replace(".json", "_eval.txt")
        fout = open(output_path, "w")
        for item in type_to_ac.keys():
            fout.write(f"Temporal category: {item}\n")
            if len(type_to_ac[item])> 0:
                fout.write(f"Avg. temporal category precision: {sum(type_to_ac[item]) / len(type_to_ac[item])}\n")
            else:
                fout.write(f"No this temporal category\n")
        for item in signal_to_ac.keys():
            fout.write(f"Temporal signal: {item}\n")
            if len(signal_to_ac[item]) > 0:
                fout.write(f"Avg. temporal signal precision: {sum(signal_to_ac[item]) / len(signal_to_ac[item])}\n")
            else:
                fout.write(f"No this temporal signal\n")

        fout.write("\n")

        fout.close()

    def inference_on_data_split(self, split, topk_answers):
        tqu = self.config["tqu"]
        benchmark = self.config["benchmark"]
        input_dir = self.config["benchmark_path"]
        output_dir = self.config["path_to_intermediate_results"]
        method_name = self.config["name"]

        dataset_file = f"{split}_input_path"
        input_path = os.path.join(input_dir, benchmark, self.config[dataset_file])
        output_dir = os.path.join(output_dir, benchmark, tqu)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_dir, f"{split}_tqu-{method_name}.json")
        """Run model on data and add predictions."""
        self.logger.info(f"TQU - Starting inference on {input_path}.")

        # open data
        with open(input_path, "r") as fp:
            data = json.load(fp)
        dataset = list()
        for instance in tqdm(data):
            dataset.append(instance)
        # model inference on given data
        self.inference_on_data(dataset, topk_answers)

        # store data
        store_json_with_mkdir(dataset, output_path)

        # log
        self.logger.info(f"TQU - Inference done on {input_path}.")
        #self.evaluate_tqu(output_path)

    def inference_on_data(self, input_data, topk_answers, sources=["kb", "text", "table", "info"]):
        """Run model on data and add predictions."""
        # model inference on given data
        for instance in tqdm(input_data):
            self.inference_on_instance(instance, topk_answers, sources)
        return input_data

    def inference_on_instance(self, instance, topk_answers, sources=["kb", "text", "table", "info"]):
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )

    def store_cache(self):
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )
