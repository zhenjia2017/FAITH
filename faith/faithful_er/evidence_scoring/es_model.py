"""
A CrossEncoder takes a sentence pair as input and outputs a label. Here, it output a continious labels 0...1 to indicate the similarity between the input pair.
It does NOT produce a sentence embedding and does NOT work for individual sentences.
"""
import math
import time
import csv
import numpy as np
import os
from faith.library.utils import get_logger
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from faith.faithful_er.evidence_scoring.dataset_es import DatasetES

from torch.utils.data import DataLoader

import torch

class ESModel(torch.nn.Module):
    def __init__(self, config):
        super(ESModel, self).__init__()
        self.config = config
        self.logger = get_logger(__name__, config)
        self.benchmark = self.config["benchmark"]
        self.data_dir = self.config["path_to_data"]
        self.bert_model_name = self.config["bert_model"]
        self.faith_or_unfaith = self.config["faith_or_unfaith"]
        self.bert_pretrain_model = self.config["bert_pretrained_model"]
        self.bert_save_model = os.path.join(self.data_dir, self.benchmark, self.faith_or_unfaith, self.bert_model_name)

        self.dataset = DatasetES(config)
        self.bert_sample_method = config["es_sample_method"]
        self.model = CrossEncoder(self.bert_pretrain_model, num_labels=1)
        self.logger.info(
            "Use pytorch device: {}".format("cuda" if torch.cuda.is_available() else "cpu")
        )

    def write_to_tsv(self, output_path, train_list):
        with open(output_path, "wt") as file:
            writer = csv.writer(file, delimiter="\t")
            header = ["ques_id", "query", "evidence", "label", "source"]
            writer.writerow(header)
            writer.writerows(train_list)

    def train(self, train_path, dev_path):
        """Train model."""
        self.logger.info(f"Starting training...")
        start = time.time()
        train_batch_size = self.config["bert_train_batch_size"]
        num_epochs = self.config["bert_num_epochs"]
        # load datasets
        self.logger.info(f"Read Train dataset")
        train_dataset, dev_dataset = self.dataset.load_data(train_path, dev_path)
        train_out_path = train_path.replace(".jsonl", f"{self.bert_sample_method}.tsv")
        self.write_to_tsv(train_out_path, train_dataset)
        dev_out_path = dev_path.replace(".jsonl", f"{self.bert_sample_method}.tsv")
        self.write_to_tsv(dev_out_path, dev_dataset)
        train_samples = []
        for item in train_dataset:
            train_samples.append(InputExample(texts=[item[1], item[2]], label=int(item[3])))
            # data augmentation
            train_samples.append(InputExample(texts=[item[2], item[1]], label=int(item[3])))
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

        self.logger.info(f"Read Dev dataset")
        dev_samples = []

        for item in dev_dataset:
            dev_samples.append(InputExample(texts=[item[1], item[2]], label=int(item[3])))

        # We add an evaluator, which evaluates the performance during training
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(
            dev_samples, name="TimeQ-dev"
        )

        # Configure the training
        warmup_steps = math.ceil(
            len(train_dataloader) * num_epochs * 0.1
        )  # 10% of train data for warm-up
        self.logger.info("Warmup-steps: {}".format(warmup_steps))

        # Train the model
        self.model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=5000,
            warmup_steps=warmup_steps,
            output_path=self.bert_save_model,
        )

        self.logger.info(f"Total time for training: {time.time() - start} seconds")

    def load(self):
        """Load model."""
        self.model = CrossEncoder(self.bert_save_model)

    def inference_top_k(self, query, evidences, max_evidence):
        """
        Retrieve the top-100 evidences among the retrieved ones,
        for the given AR.
        """
        start = time.time()
        if not evidences:
            return evidences
        mapping = {}

        query_evidence_pairs = list()
        # query = truecase.get_true_case(query)
        for evidence in evidences:
            # remove noise in evidence texts
            evidence_text = evidence["evidence_text"].replace("\n", " ").replace("\t", " ")
            if evidence_text not in mapping:
                mapping[evidence_text] = list()
            mapping[evidence_text].append(evidence)
        # Compute embedding for both lists
        evidence_texts = list(mapping.keys())
        top_evidences = list()
        for evidence_text in evidence_texts:
            query_evidence_pairs.append([query, evidence_text])

        similarity_scores = self.model.predict(query_evidence_pairs)
        sim_scores_argsort = reversed(np.argsort(similarity_scores))
        scored_evidences = [query_evidence_pairs[idx] for idx in sim_scores_argsort][
            : max_evidence
        ]
        for query, evidence_text in scored_evidences:
            top_evidences += mapping[evidence_text]
        self.logger.info(f"Total time for inference: {time.time() - start} seconds")

        return top_evidences
