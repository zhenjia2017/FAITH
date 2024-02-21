import os
import torch
import transformers

from pathlib import Path
from faith.library.utils import get_logger
import faith.heterogeneous_answering.seq2seq_answering.dataset_seq2seq_answering as dataset


class Seq2SeqAnsweringModel(torch.nn.Module):
    def __init__(self, config):
        super(Seq2SeqAnsweringModel, self).__init__()
        self.config = config
        self.logger = get_logger(__name__, config)
        self.fer = self.config["fer"]
        self.ha = self.config["ha"]
        self.evs_max_evidences = self.config["evs_max_evidences"]

        self.data_path = self.config["path_to_data"]
        self.benchmark = self.config["benchmark"]
        self.faith_or_unfaith = self.config["faith_or_unfaith"]
        self.ha_model_dir = os.path.join(self.data_path, self.benchmark, self.faith_or_unfaith,
                     self.ha)
        Path(self.ha_model_dir).mkdir(parents=True, exist_ok=True)
        self.ha_model_path = os.path.join(self.ha_model_dir, f"top{self.evs_max_evidences}.bin")

        # select model architecture
        if config["ha_architecture"] == "BART":
            self.model = transformers.BartForConditionalGeneration.from_pretrained(
                "facebook/bart-base", ignore_mismatched_sizes=True
            )
            self.tokenizer = transformers.BartTokenizerFast.from_pretrained("facebook/bart-base")
            self.tokenizer.add_tokens(["</tsf>", "</e>"])
            self.model.resize_token_embeddings(len(self.tokenizer))

        elif config["ha_architecture"] == "T5":
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(
                "t5-base", ignore_mismatched_sizes=True
            )
            self.tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-base")
            self.tokenizer.add_tokens(["</tsf>", "</e>"])
            self.model.resize_token_embeddings(len(self.tokenizer))

        elif config["ha_architecture"] == "T5_small":
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(
                "t5-small", ignore_mismatched_sizes=True
            )
            self.tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-small")
            self.tokenizer.add_tokens(["</tsf>", "</e>"])
            self.model.resize_token_embeddings(len(self.tokenizer))

        else:
            raise Exception(
                "Unknown architecture for Seq2SeqAnsweringModel module specified in config."
            )

    def set_eval_mode(self):
        """Set model to eval mode."""
        self.model.eval()

    def save(self):
        """Save model."""
        # create dir if not exists
        model_dir = os.path.dirname(self.ha_model_path)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.ha_model_path)
        # self.tokenizer.save_pretrained(self.config["ha_tokenizer_path"])

    def load(self):
        """Load model."""
        if torch.cuda.is_available():
            state_dict = torch.load(self.ha_model_path)
        else:
            state_dict = torch.load(self.ha_model_path, torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def train(self, train_path, dev_path):
        """Train model."""
        architecture = self.config["ha_architecture"]
        # load datasets
        train_dataset = dataset.DatasetSeq2SeqAnswering(self.config, self.tokenizer, train_path)
        dev_dataset = dataset.DatasetSeq2SeqAnswering(self.config, self.tokenizer, dev_path)
        self.logger.info(f"Length train data: {len(train_dataset)}")
        self.logger.info(f"Length dev data: {len(dev_dataset)}")
        # arguments for training
        training_args = transformers.Seq2SeqTrainingArguments(
            output_dir=f"faith/heterogeneous_answering/seq2seq_answering/results/{architecture}-{self.fer}-top{self.evs_max_evidences}",  # output directory
            num_train_epochs=self.config["ha_num_train_epochs"],  # total number of training epochs
            per_device_train_batch_size=self.config[
                "ha_per_device_train_batch_size"
            ],  # batch size per device during training
            per_device_eval_batch_size=self.config[
                "ha_per_device_eval_batch_size"
            ],  # batch size for evaluation
            warmup_steps=self.config[
                "ha_warmup_steps"
            ],  # number of warmup steps for learning rate scheduler
            weight_decay=self.config["ha_weight_decay"],  # strength of weight decay
            logging_dir=f"faith/heterogeneous_answering/seq2seq_answering/logs/{architecture}-{self.fer}-top{self.evs_max_evidences}",  # directory for storing logs
            logging_steps=1000,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end="True"
            # predict_with_generate=True
        )
        # create the object for training
        trainer = transformers.Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
        )
        # training progress
        trainer.train()
        # store model
        self.save()

    def inference(self, input_text):
        """Run the model on the given input."""
        # encode
        input_encodings = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=self.config["ha_max_input_length"],
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            input_encodings = input_encodings.to(torch.device("cuda"))
        # generate
        output = self.model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            # no_repeat_ngram_size=self.config["ha_no_repeat_ngram_size"],
            # num_beams=self.config["ha_num_beams"],
            # early_stopping=self.config["ha_early_stopping"],
        )
        # decoding
        generated_answer = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        return generated_answer

    def inference_test(self, input_text):
        """Run the model on the given input."""
        # encode
        input_encodings = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=self.config["ha_max_input_length"],
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            input_encodings = input_encodings.to(torch.device("cuda"))
        # generate
        beam_outputs = self.model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            no_repeat_ngram_size=self.config["ha_no_repeat_ngram_size"],
            num_beams=self.config["ha_num_beams"],
        )
        # decoding
        generated_answers = list()
        for i, beam_output in enumerate(beam_outputs):
            generated_answer = self.tokenizer.decode(
                beam_output,
                skip_special_tokens=True,
            )
            generated_answers.append(generated_answer)
        return generated_answers

    def inference_on_batch(self, inputs):
        """Run the model on the given inputs (batch)."""
        # encode inputs
        input_encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config["ha_max_input_length"],
            return_tensors="pt",
        )
        # generation
        summary_ids = self.model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            no_repeat_ngram_size=self.config["sr_no_repeat_ngram_size"],
            num_beams=self.config["ha_num_beams"],
            early_stopping=self.config["ha_early_stopping"],
        )
        # decoding
        output = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for g in summary_ids
        ]
        return output
