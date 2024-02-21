import os
import sys
import yaml
import json
import logging
from pathlib import Path

def get_config(path):
    """Load the config dict from the given .yml file."""
    with open(path, "r") as fp:
        config = yaml.safe_load(fp)
    return config

def store_json_with_mkdir(data, output_path, indent=True):
    """Store the JSON data in the given path."""
    # create path if not exists
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        fp.write(json.dumps(data, indent=4))


def store_jsonl_with_mkdir(data, output_path, indent=False):
    """Store the JSON data in the given path."""
    # create path if not exists
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        for inst in data:
            fp.write(json.dumps(inst, indent=4 if indent else None))
            fp.write("\n")


def tsf_dic_to_string(tsf):
    if isinstance(tsf, dict):
        entity = tsf["entity"]
        relation = tsf["relation"]
        ans_type = tsf["answer_type"]
        temporal_signal = tsf["temporal_signal"]
        category = tsf["category"]
        query = (
            f"{entity} {relation} {ans_type} {temporal_signal} {category}"
        )
        if tsf["temporal_value"]:
            for item in tsf["temporal_value"]:
                if isinstance(item, int):
                    query += " " + str(item)
                elif isinstance(item, list):
                    query += " " + item[0] + " " + item[1]
        return query
    elif "||" in tsf:
        # remove delimiter from SR
        slots = tsf.split("||")
        # "entity || relation || answer_type || _ || Temp.Ans || _ "
        slots[3] = slots[3].replace("_", "")
        slots[5] = slots[5].replace("_", "").replace("T00:00:00Z", "")
        query = " ".join([item.strip() for item in slots])
        return query
    else:
        return tsf


def get_logger(mod_name, config):
    """Get a logger instance for the given module name."""
    # create logger
    logger = logging.getLogger(mod_name)
    # add handler and format
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # set log level
    log_level = config["log_level"]
    logger.setLevel(getattr(logging, log_level))
    return logger


def get_result_logger(config):
    """Get a logger instance for the given module name."""
    # create logger
    logger = logging.getLogger("result_logger")
    # add handler and format
    method_name = config["name"]
    benchmark = config["benchmark"]
    result_file = f"_results/{benchmark}/{method_name}.res"
    result_dir = os.path.dirname(result_file)
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(result_file)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # set log level
    logger.setLevel("INFO")
    return logger


def print_dict(python_dict):
    """Print python dict as json-string."""
    json_string = json.dumps(python_dict)
    print(json_string)


def print_verbose(config, string):
    """Print the given string if verbose is set."""
    if config["verbose"]:
        print(str(string))


def extract_mapping_incomplete_complete(data_paths):
    """
    Extract mapping from incomplete questions to complete
    questions for all follow-up questions.
    """
    mapping_incomplete_to_complete = dict()
    for data_path in data_paths:
        with open(data_path, "r") as fp:
            dataset = json.load(fp)

        for conversation in dataset:
            for turn in conversation["questions"]:
                if turn["turn"] == 0:
                    continue
                question = turn["question"]
                completed = turn["completed"]
                mapping_incomplete_to_complete[question] = completed
    return mapping_incomplete_to_complete

