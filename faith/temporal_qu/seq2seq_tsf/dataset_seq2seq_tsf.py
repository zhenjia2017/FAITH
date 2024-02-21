import json
import torch

bad_questions = []

def input_to_text(instance):
    """
    Transform the question into the input text.
    """
    question = instance["question"]
    return question

def output_to_text(instance, tsf_delimiter):
    """
    Transform the given silver temporal structure form to text.
    The (recursive) list data structure is resolved and flattened.
    """
    silver_tsf = instance["silver_tsf"]
    # silver_tsf is a dictionary and the keys are as follows.
    # "entity": question entity,
    # "relation": question relation,
    # "answer_type": answer type,
    # "temporal_signal": temporal signal,
    # "categorization": temporal category
    entities = silver_tsf["entity"]
    relation = silver_tsf["relation"]
    answer_type = silver_tsf["answer_type"]
    temporal_signal = silver_tsf["temporal_signal"]
    categorization = silver_tsf["categorization"]

    entities = " ".join(entities).strip()
    ans_type = answer_type.strip() if answer_type else ""
    temp_signal = temporal_signal.strip() if temporal_signal else ""
    temporal_type = "implicit" if "Implicit" in categorization else "none"

    tsf_text = f"{entities}{tsf_delimiter}{relation}{tsf_delimiter}{ans_type}{tsf_delimiter}{temp_signal}{tsf_delimiter}{temporal_type}"

    # remove whitespaces in TSF
    while "  " in tsf_text:
        tsf_text = tsf_text.replace("  ", " ")
    tsf_text = tsf_text.replace(" , ", ", ")
    tsf_text = tsf_text.strip()
    return tsf_text


class DatasetSeq2SeqTSF(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, path):
        self.config = config
        self.tokenizer = tokenizer
        self.tsf_delimiter = config["tsf_delimiter"]

        input_encodings, output_encodings, dataset_length = self._load_data(path)
        self.input_encodings = input_encodings
        self.output_encodings = output_encodings
        self.dataset_length = dataset_length

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        labels = self.output_encodings["input_ids"][idx]
        item = {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": labels,
        }
        return item

    def __len__(self):
        return self.dataset_length

    def _load_data(self, path):
        """
        Opens the file, and loads the data into a format that can be put into the model.
        The input dataset should be annotated using the TSFAnnotator class in tsf_annotation.py.
        """
        # open data
        with open(path, "r") as fp:
            dataset = json.load(fp)

        inputs = list()
        outputs = list()

        for instance in dataset:
            # skip examples for which no gold TSF was found
            if not instance["silver_tsf"]:
                continue

            inputs.append(input_to_text(instance))
            outputs.append(output_to_text(instance, self.tsf_delimiter))

        # encode
        input_encodings = self.tokenizer(
            inputs, padding=True, truncation=True, max_length=self.config["tsf_max_input_length"]
        )
        output_encodings = self.tokenizer(
            outputs, padding=True, truncation=True, max_length=self.config["tsf_max_input_length"]
        )
        dataset_length = len(inputs)

        return input_encodings, output_encodings, dataset_length
