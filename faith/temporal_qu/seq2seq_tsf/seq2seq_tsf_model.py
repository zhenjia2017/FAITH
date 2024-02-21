import os
import torch
import transformers

from pathlib import Path
import faith.temporal_qu.seq2seq_tsf.dataset_seq2seq_tsf as dataset


class Seq2SeqTSFModel(torch.nn.Module):
    def __init__(self, config):
        super(Seq2SeqTSFModel, self).__init__()
        self.config = config
        self.tsf_delimiter = self.config["tsf_delimiter"]
        self.benchmark = self.config["benchmark"]
        self.data_dir = self.config["path_to_data"]
        self.tsf_model = self.config["tsf_model"]
        self.tsf_model_path = os.path.join(self.data_dir, self.benchmark, self.tsf_model)
        # select model architecture
        if config["tsf_architecture"] == "BART":
            self.model = transformers.BartForConditionalGeneration.from_pretrained(
                "facebook/bart-base"
            )
            self.tokenizer = transformers.BartTokenizerFast.from_pretrained("facebook/bart-base")
        elif config["tsf_architecture"] == "T5":
            self.model = transformers.T5ForConditionalGeneration.from_pretrained("t5-base")
            self.tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-base")
        else:
            raise Exception(
                "Unknown architecture for TSF module specified in config: currently, only T5-base (=T5) and BART-base (=BART) are supported."
            )

    def set_eval_mode(self):
        """Set model to eval mode."""
        self.model.eval()

    def save(self):
        """Save model."""
        model_path = self.tsf_model_path
        # create dir if not exists
        model_dir = os.path.dirname(model_path)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def load(self):
        """Load model."""
        print(f"torch.cuda.is_available(),{torch.cuda.is_available()}")
        if torch.cuda.is_available():
            state_dict = torch.load(self.tsf_model_path)
        else:
            state_dict = torch.load(self.tsf_model_path, torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def train(self, train_path, dev_path):
        """Train model."""
        # load datasets
        train_dataset = dataset.DatasetSeq2SeqTSF(self.config, self.tokenizer, train_path)
        dev_dataset = dataset.DatasetSeq2SeqTSF(self.config, self.tokenizer, dev_path)
        # arguments for training
        training_args = transformers.Seq2SeqTrainingArguments(
            output_dir="faith/temporal_qu/seq2seq_tsf/results",  # output directory
            num_train_epochs=self.config["tsf_num_train_epochs"],  # total number of training epochs
            per_device_train_batch_size=self.config[
                "tsf_per_device_train_batch_size"
            ],  # batch size per device during training
            per_device_eval_batch_size=self.config[
                "tsf_per_device_eval_batch_size"
            ],  # batch size for evaluation
            warmup_steps=self.config[
                "tsf_warmup_steps"
            ],  # number of warmup steps for learning rate scheduler
            weight_decay=self.config["tsf_weight_decay"],  # strength of weight decay
            logging_dir="faith/temporal_qu/seq2seq_tsf/logs",  # directory for storing logs
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

    def inference_top_1(self, input):
        """Run the model on the given input."""
        # encode
        input_encodings = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            max_length=self.config["tsf_max_input_length"],
            return_tensors="pt",
        )
        print(f"torch.cuda.is_available(),{torch.cuda.is_available()}")
        if torch.cuda.is_available():
            input_encodings = input_encodings.to(torch.device("cuda"))

        # generate
        output = self.model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            no_repeat_ngram_size=self.config["tsf_no_repeat_ngram_size"],
            num_beams=self.config["tsf_num_beams"],
            early_stopping=self.config["tsf_early_stopping"],
            max_length=self.config["tsf_max_length"],
        )

        # decoding
        tsfs = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        tsf = tsfs[0]
        return tsf

    def inference_top_k(self, input):
        """Run the model on the given input."""
        # encode
        input_encodings = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            max_length=self.config["tsf_max_input_length"],
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            input_encodings = input_encodings.to(torch.device("cuda"))
        # generate
        outputs = self.model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            no_repeat_ngram_size=self.config["tsf_no_repeat_ngram_size"],
            num_beams=self.config["tsf_num_beams"],
            early_stopping=self.config["tsf_early_stopping"],
            num_return_sequences=self.config["tsf_k"],
            max_length=self.config["tsf_max_length"],
        )

        tsfs = [
            self.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for output in outputs
        ]
        return tsfs

    def inference_on_batch(self, inputs):
        """Run the model on the given inputs (batch)."""
        # encode inputs
        input_encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config["tsf_max_input_length"],
            return_tensors="pt",
        )
        # generation
        summary_ids = self.model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            no_repeat_ngram_size=self.config["tsf_no_repeat_ngram_size"],
            num_beams=self.config["tsf_num_beams"],
            early_stopping=self.config["tsf_early_stopping"],
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
