import os
import sys
import torch
import json

from tqdm import tqdm
from faith.library.utils import get_config, get_logger, store_json_with_mkdir
from faith.temporal_qu.seq2seq_tsf.seq2seq_tsf_model import Seq2SeqTSFModel

"""
The script is for fine-tuning the BART model.
The script is for generating the TSF using the BART model.
"""

class Seq2SeqTSFModule:
    def __init__(self, config):
        """Initialize TSF module."""
        self.config = config
        self.logger = get_logger(__name__, config)

        # create model
        self.tsf_model = Seq2SeqTSFModel(config)
        self.model_loaded = False

        self.tsf_delimiter = config["tsf_delimiter"]

    def train(self):
        """Train the model on silver TSF data."""
        # train model
        self.logger.info(f"Starting training...")
        benchmark = self.config["benchmark"]
        path_to_data = self.config["path_to_data"]
        data_dir = os.path.join(path_to_data, benchmark, self.config["tsf_annotated_path"])

        tsf_train_data = self.config["tsf_train"]
        tsf_dev_data = self.config["tsf_dev"]

        train_path = os.path.join(data_dir, tsf_train_data)
        dev_path = os.path.join(data_dir, tsf_dev_data)

        self.tsf_model.train(train_path, dev_path)
        self.logger.info(f"Finished training.")

    def inference_on_instance(self, instance):
        """Run inference on a single question."""
        # load TSF model (if required)
        self._load()
        with torch.no_grad():
            if "question" not in instance and "Question" in instance:
                instance["question"] = instance["Question"]
            instance["structured_temporal_form"] = self._inference(instance["question"])
        return instance

    def inference_on_question(self, question):
        """Run inference on a single question."""
        return self.inference_on_instance({"question": question})["structured_temporal_form"]

    def _inference(self, question):
        def _normalize_input(_input):
            return _input.replace(",", " ")

        def _normalize_tsf(tsf):
            # drop type ("hallucination" is desired there)
            tsf_slots = tsf.split(self.tsf_delimiter, 3)
            tsf = " ".join((tsf_slots[:1] + [tsf_slots[3]]))
            tsf = tsf.replace(",", " ").replace(self.tsf_delimiter.strip(), " ")
            return tsf

        def _hallucination(input_words, tsf_words):
            """Check if the model hallucinated (except for type)."""
            bools = [word in input_words for word in tsf_words]
            if False in bools:
                return True
            return False

        # try to avoid hallucination: get top-k TSFs, and take first
        # that only output words that are there in input (except for type)
        # since we generate signal and question category, tsf_avoid_hallucination should be always false
        if self.config.get("tsf_avoid_hallucination"):
            tsfs = self.tsf_model.inference_top_k(question)
            tsfs = [self._format_tsf(tsf) for tsf in tsfs]

            # get input words
            input_words = _normalize_input(question).split()
            for tsf in tsfs:
                tsf_words = _normalize_tsf(tsf).split()
                # return first SR without hallucination
                if not _hallucination(input_words, tsf_words):
                    return tsf

            # if hallucination is there in all TSF candidates, return the top-ranked
            return tsfs[0]
        # default inference
        else:
            tsf = self.tsf_model.inference_top_1(question)
            tsf = self._format_tsf(tsf)
            return tsf

    def _format_tsf(self, tsf, output_format="sequence"):
        """Make sure the seq2seq generated TSF has 4 delimiters and 5 slots.
            f"{entities}{tsf_delimiter}{relation}{tsf_delimiter}{ans_type}{tsf_delimiter}{temp_signal}{tsf_delimiter}{temp_category}"
        """
        if output_format == "sequence":
            slots = tsf.split(self.tsf_delimiter.strip(), 4)
            if len(slots) < 5:
                # in case there are still less than 4 slots
                slots = slots + (5 - len(slots)) * [""]
            tsf = self.tsf_delimiter.join(slots)
        else:
            tsf = eval(tsf)
        return tsf

    def _load(self):
        """Load the TSF model."""
        # only load if not already done so
        if not self.model_loaded:
            self.tsf_model.load()
            self.tsf_model.set_eval_mode()
            self.model_loaded = True


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception(
            "Usage: python faith/temporal_qu/seq2seq_tsf/seq2seq_tsf_module.py --<FUNCTION> <PATH_TO_CONFIG>"
        )

    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    # train: train model
    if function == "--train":
        srm = Seq2SeqTSFModule(config)
        srm.train()

    # inference: add predictions to data
    elif function == "--example":
        # load config
        example = "what awards were gladys knight & the pips nominated for during wwii"
        srm = Seq2SeqTSFModule(config)
        res = srm.inference_on_question(example)
        print(res)
