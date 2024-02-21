import os
import sys
import torch
import json
from tqdm import tqdm
from faith.library.utils import get_config, get_logger
from faith.temporal_qu.implicit_resolver.seq2seq_iques.seq2seq_iques_model import Seq2SeqIQUESModel

"""
The script is for fine-tuning the BART model.
The script is for generating the sequence of intermediate questions and answer type (date or time interval) using the BART model.
"""

class Seq2SeqIQUESModule:
    def __init__(self, config):
        """Initialize iQUES module."""
        self.config = config
        self.logger = get_logger(__name__, config)

        # create model
        self.iques_model = Seq2SeqIQUESModel(config)
        self.model_loaded = False

    def train(self):
        """Train the model on the generated intermediate questions data."""
        # train model
        self.logger.info(f"Starting training...")
        benchmark = self.config["benchmark"]
        path_to_data = self.config["path_to_data"]
        data_dir = os.path.join(path_to_data, benchmark, self.config["intermediate_question_dataset_path"])
        train_file = self.config["intermediate_question_train"]
        dev_file = self.config["intermediate_question_dev"]
        #intermediate question dataset generated based on instructGPT
        train_path = os.path.join(data_dir, train_file)
        dev_path = os.path.join(data_dir, dev_file)
        self.iques_model.train(train_path, dev_path)
        self.logger.info(f"Finished training.")

    def inference_on_instance(self, instance):
        """Run inference on a single question."""
        # load iques model (if required)
        self._load()
        with torch.no_grad():
            question = instance["Question"]
            try:
                result = self._inference(question)
                instance["generated_iquestion"] = result
                self.logger.info(f"Generate question for {question} and the result is: {result}")
            except:
                self.logger.info(f"Fail to generate question for: {question}")
        return instance

    def inference_on_question(self, question):
        """Run inference on a single question."""
        return self.inference_on_instance({"Question": question})["generated_iquestion"]

    def _inference(self, question):
        iques = self.iques_model.inference_top_1(question)
        return iques

    def _load(self):
        """Load the iques model."""
        # only load if not already done so
        if not self.model_loaded:
            self.iques_model.load()
            self.iques_model.set_eval_mode()
            self.model_loaded = True


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise Exception(
            "Usage: python faith/temporal_qu/seq2seq_iques/dataset_creation/seq2seq_iques_module.py --<FUNCTION> <PATH_TO_CONFIG>"
        )

    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    # train: train model
    if function == "--train":
        srm = Seq2SeqIQUESModule(config)
        srm.train()

    # inference: add predictions to data
    elif function == "--inference":
        # load config
        # open data
        srm = Seq2SeqIQUESModule(config)
        benchmark_dir = config["benchmark_path"]
        benchmark = config["benchmark"]
        test_path = os.path.join(benchmark_dir, benchmark, config["test_input_path"])
        with open(test_path, "r") as fp:
            data = json.load(fp)
        instances = []
        # process data
        for instance in tqdm(data):
            srm.inference_on_instance(instance)
            instances.append(instance)
        with open(os.path.join(test_path, f'gpt_annotate_generate_question_test_inference.json'), "w") as fp:
            fp.write(json.dumps(instances, indent=4))

    # inference: add predictions to data
    elif function == "--example":
        # load config
        example = "what awards were gladys knight & the pips nominated for during wwii"
        srm = Seq2SeqIQUESModule(config)
        res = srm.inference_on_question(example)
        print(res)
