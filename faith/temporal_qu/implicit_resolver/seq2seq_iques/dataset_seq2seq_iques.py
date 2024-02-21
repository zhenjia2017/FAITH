import json
import torch

def input_to_text(instance):
    """
    Transform the question into the input text.
    """
    question = instance["Question"]
    return question

def output_to_text(instance, delimiter):
    """
    Transform the given generated question and answer type to text.
    The (recursive) list data structure is resolved and flattened.
    """
    generated_question = instance["silver_generated_question"]

    intermiediate_question, ans_type = generated_question

    # create ar text
    iques_text = f"{intermiediate_question}{delimiter}{ans_type}"

    # remove multiple whitespaces in question
    while "  " in iques_text:
        iques_text = iques_text.replace("  ", " ")
    iques_text = iques_text.replace(" , ", ", ")
    iques_text = iques_text.strip()
    return iques_text


class DatasetSeq2SeqIQUES(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, path):
        self.config = config
        self.tokenizer = tokenizer
        self.delimiter = config["iques_delimiter"]

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
        Opens the file, and loads the data into
        a format that can be put into the model.

        The input dataset is generated using the methods in the package "dataset_creation".

        The question is given as input.
        """
        # open data
        with open(path, "r") as fp:
            dataset = json.load(fp)

        inputs = list()
        outputs = list()

        for instance in dataset:
            # skip examples for which no gold generated question was found
            if not instance["silver_generated_question"]:
                continue

            inputs.append(input_to_text(instance))
            outputs.append(output_to_text(instance, self.delimiter))

        # encode
        input_encodings = self.tokenizer(
            inputs, padding=True, truncation=True, max_length=self.config["iques_max_input_length"]
        )
        output_encodings = self.tokenizer(
            outputs, padding=True, truncation=True, max_length=self.config["iques_max_input_length"]
        )
        dataset_length = len(inputs)

        return input_encodings, output_encodings, dataset_length
