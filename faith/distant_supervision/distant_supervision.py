import os
import sys
import json
import pickle

from tqdm import tqdm
from pathlib import Path

from clocq.CLOCQ import CLOCQ
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient

from faith.library.utils import get_config, get_logger

from faith.distant_supervision.tsf_annotator import TSFAnnotator

class DistantSupervision:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)

        # initialize clocq
        if config["clocq_use_api"]:
            self.clocq = CLOCQInterfaceClient(host=config["clocq_host"], port=config["clocq_port"])
        else:
            self.clocq = CLOCQ(dev=True)

        # initialize annotators
        self.tsf_annotator = TSFAnnotator(self.clocq, config)

        #  open labels
        #labels_path = config["path_to_labels"]
        # with open(labels_path, "rb") as fp:
        #     self.labels_dict = pickle.load(fp)

    def process_dataset(self, dataset_path, output_path):
        """
        Annotate the given dataset and store the output in the specified path.
        """
        with open(dataset_path, "r") as fp:
            dataset = json.load(fp)
        # process data
        tsf_count = 0
        for instance in tqdm(dataset):
            # annotate data
            success = self.tsf_annotator.process_instance(instance)
            if success:
                tsf_count += 1

        # log
        self.logger.info(f"Done with DS on: {dataset_path}")
        self.logger.info(f"\t#TSFs extracted: {tsf_count}")
        self.logger.info(f"\t#Total questions: {len(dataset)}")

        # store annotated dataset
        with open(output_path, "w") as fp:
            fp.write(json.dumps(dataset, indent=4))

#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception(
            "Usage: python faith/distant_supervision/distant_supervision.py --function <PATH_TO_CONFIG>"
        )

    # load options and config
    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)
    benchmark_path = config["benchmark_path"]
    benchmark = config["benchmark"]
    path_to_data = config["path_to_data"]
    output_dir = os.path.join(path_to_data, benchmark, config["tsf_annotated_path"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # inference using predicted answers
    if function == "--train":
        # create annotator
        annotator = DistantSupervision(config)
        input_path = os.path.join(benchmark_path, benchmark, config["train_input_path"])
        output_path = os.path.join(output_dir, config["tsf_train"])
        annotator.process_dataset(input_path, output_path)

    elif function == "--dev":
        # create annotator
        annotator = DistantSupervision(config)
        input_path = os.path.join(benchmark_path, benchmark, config["dev_input_path"])
        output_path = os.path.join(output_dir, config["tsf_dev"])
        annotator.process_dataset(input_path, output_path)




