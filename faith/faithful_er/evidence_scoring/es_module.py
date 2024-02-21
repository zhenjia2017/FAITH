import time
import os
from faith.library.utils import get_logger
from faith.faithful_er.evidence_scoring.es_model import ESModel
from faith.faithful_er.faithful_evidence_retrieval import FaithfulEvidenceRetrieval

class ESModule(FaithfulEvidenceRetrieval):
    def __init__(self, config):
        self.config = config
        self.faith = self.config["faith_or_unfaith"]
        self.benchmark = self.config["benchmark"]
        self.logger = get_logger(__name__, config)
        # create model
        self.bert_model = ESModel(config)
        self.model_loaded = False

    def train(self, sources=["kb", "text", "table", "info"]) -> object:
        """Train the model on evidence retrieval data."""
        input_dir = self.config["path_to_intermediate_results"]
        tqu = self.config["tqu"]
        method_name = self.config["name"]
        fer = self.config["fer"]
        start = time.time()
        sources_str = "_".join(sources)
        if self.faith == "faith":
            train_path = os.path.join(input_dir, self.benchmark, tqu, fer, sources_str,
                                      f"train_erp-{method_name}.jsonl")
            dev_path = os.path.join(input_dir, self.benchmark, tqu, fer, sources_str, f"dev_erp-{method_name}.jsonl")
        else:
            train_path = os.path.join(input_dir, self.benchmark, tqu, fer, sources_str, f"train_er-{method_name}.jsonl")
            dev_path = os.path.join(input_dir, self.benchmark, tqu, fer, sources_str, f"dev_er-{method_name}.jsonl")
        self.logger.info(f"Starting training...")
        self.bert_model.train(train_path, dev_path)
        self.logger.info(f"Finished training.")
        running_time = time.time() - start
        self.logger.info(f"Training time of the scoring model is {running_time}")

    def get_top_evidences(self, query, evidences, max_evidence):
        """Run inference on a single question."""
        # load Bert model (if required)
        self._load()
        top_evidences = self.bert_model.inference_top_k(query, evidences, max_evidence)
        return top_evidences

    def _load(self):
        """Load the bert_model."""
        # only load if not already done so
        if not self.model_loaded:
            self.bert_model.load()
            self.model_loaded = True
