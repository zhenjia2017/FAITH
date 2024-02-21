import json
from tqdm import tqdm
import random
from rank_bm25 import BM25Okapi
import csv

from torch.utils.data import Dataset
from faith.library.utils import get_config, get_logger

class DatasetES(Dataset):
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.pos_sample_num = config["bert_max_pos_evidences_per_source"]
        self.neg_sample_num = config["bert_max_neg_evidences"]
        self.bert_sample_method = config["es_sample_method"]
        self.stop_words_file = config["path_to_stopwords"]
        with open(self.stop_words_file, "r") as fp:
            self.stopwords = fp.read().split("\n")

    # remove evidences from negative when the evidences are in both positive and negative
    def remove_duplicate_evidence(self, positive, negative):
        for item in negative:
            if item in list(positive.values()):
                negative.remove(item)

    def bm25rank(self, evidences):
        """
        Retrieve the top-100 evidences among the retrieved ones,
        for the given AR.
        """

        def tokenize(string):
            """Function to tokenize string (word-level)."""
            string = string.replace(",", " ")
            string = string.strip()
            return [word.lower() for word in string.split() if not word in self.stopwords]

        if not evidences:
            return evidences

        # tokenize
        # [query, evidence_text, 1]
        query = evidences[0][0]
        evidence_texts = [evidence[1] for evidence in evidences]
        mapping = {
            " ".join(tokenize(evidence_text)): evidence_text for evidence_text in evidence_texts
        }
        tokenized_tsf = tokenize(query)

        # create corpus
        tokenized_corpus = [tokenize(evidence_text) for evidence_text in evidence_texts]
        bm25_module = BM25Okapi(tokenized_corpus)

        # scoring
        scores = bm25_module.get_scores(tokenized_tsf)

        ranked_indices = sorted(range(len(tokenized_corpus)), key=lambda i: scores[i], reverse=True)

        scored_evidences = [
            mapping[" ".join(tokenized_corpus[index])] for i, index in enumerate(ranked_indices)
        ]
        return scored_evidences

    def answer_presence(self, answer_candidate_ids, gold_answer_ids):
        """Check whether the given evidence has any of the answers."""
        for answer_candidate_id in answer_candidate_ids:
            # check for year in case the item is a timestamp

            # check if answering candidate
            # normalize
            answer_candidate_id = (
                answer_candidate_id.lower().strip().replace('"', "").replace("+", "")
            )
            gold_answer_ids = [
                answer.lower().strip().replace('"', "").replace("+", "")
                for answer in gold_answer_ids
            ]

            # perform check
            if answer_candidate_id in gold_answer_ids:
                return True
        # no match found
        return False

    def BM25_samle(self, positives, pos_num):
        scored_evidences = self.bm25rank(positives)
        positive_samples = []
        # [query, evidence_text, 1]
        for item in positives:
            if item[1] == scored_evidences[0] and len(positive_samples) < pos_num:
                positive_samples.append(item)
        return positive_samples

    # construct training dataset for bert model
    def process_dataset(self, dataset_path, sources=["kb", "text", "table", "info"]):
        # process data
        training_dataset = list()
        with open(dataset_path, "r") as fp:
            for line in tqdm(fp):
                try:
                    instance = json.loads(line)
                except:
                    print ("Load json error!")
                    continue

                ques_id = instance["Id"]
                positive_evidences = dict()
                negative_evidences = list()

                gt = set([item["id"] for item in instance["answers"]])
                tsf = instance["structured_temporal_form"]
                if isinstance(tsf, dict):
                    if "entity" in tsf and "relation" in tsf:
                        entity = tsf["entity"].strip()
                        relation = tsf["relation"].strip()
                        answer_type = tsf["answer_type"].strip()
                        # for keeping the consistency with the clocq_er, we add answer type for ranking
                        query = f"{entity}{' '}{relation}{' '}{answer_type}"
                else:
                    query = tsf

                has_answer = instance["answer_presence"]

                # skip the instance without gold answer
                if not has_answer:
                    continue

                evidence_dic = {}
                for evidence in instance["candidate_evidences"]:
                    evidence_text = evidence["evidence_text"].replace("\n", " ").replace("\t", " ")
                    source = evidence["source"]
                    key = evidence_text + " || " + source
                    if key not in evidence_dic:
                        evidence_dic[key] = set()
                    evidence_dic[key] |= set([item["id"] for item in evidence["wikidata_entities"]])

                    # this is already fixed in the new dataset
                    if evidence["tempinfo"]:
                        for item in evidence["tempinfo"][1]:
                            evidence_dic[key] |= {item[1]}

                for key in evidence_dic.keys():
                    source = key.split(" || ")[1]
                    evidence_text = key.split(" || ")[0]
                    if self.answer_presence(evidence_dic[key], gt):
                        if source in sources:
                            if source not in positive_evidences:
                                positive_evidences[source] = []
                                # the label of positive evidence is integar 1
                            positive_evidences[source].append(
                                [ques_id, query, evidence_text, 1, source]
                            )
                    else:
                        # record the negative evidences of required sources
                        if source in sources:
                            # the label of positive evidence is integar 0
                            negative_evidences.append([ques_id, query, evidence_text, 0, source])

                if len(positive_evidences.values()) > 0 and len(negative_evidences) > 0:
                    # sample one positive evidence from each source
                    for source in positive_evidences.keys():
                        # sample top-1 positive evidence from BM25 ranking result for each source
                        positive_evidence_items = positive_evidences[source]
                        if self.bert_sample_method == "bm25_sample":
                            training_dataset += self.BM25_samle(
                                positive_evidence_items, self.pos_sample_num
                            )
                        elif self.bert_sample_method == "random_sample":
                            training_dataset += random.sample(
                                positive_evidence_items, self.pos_sample_num
                            )

                    # sample negative evidences
                    sample_num = self.neg_sample_num
                    if len(negative_evidences) < self.neg_sample_num:
                        sample_num = len(negative_evidences)

                    # random sample negative
                    training_dataset += random.sample(negative_evidences, sample_num)

        return training_dataset

    def load_data(self, train_path, dev_path):
        training_dataset = self.process_dataset(train_path)
        dev_dataset = self.process_dataset(dev_path)
        return training_dataset, dev_dataset

    def write_to_tsv(self, output_path, train_list):
        with open(output_path, "wt") as file:
            writer = csv.writer(file, delimiter="\t")
            header = ["ques_id", "query", "evidence", "label", "source"]
            writer.writerow(header)
            writer.writerows(train_list)


