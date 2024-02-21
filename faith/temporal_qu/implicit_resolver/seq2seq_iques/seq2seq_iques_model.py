import os
import torch
import transformers
import faith.temporal_qu.implicit_resolver.seq2seq_iques.dataset_seq2seq_iques as dataset

class Seq2SeqIQUESModel(torch.nn.Module):
    def __init__(self, config):
        super(Seq2SeqIQUESModel, self).__init__()
        self.config = config
        self.benchmark = self.config["benchmark"]
        self.data_dir = self.config["path_to_data"]
        self.iques_model = self.config["iques_model"]
        self.iques_model_path = os.path.join(self.data_dir, self.benchmark, self.iques_model)
        # select model architecture
        if config["iques_architecture"] == "BART":
            self.model = transformers.BartForConditionalGeneration.from_pretrained("facebook/bart-base")
            self.tokenizer = transformers.BartTokenizerFast.from_pretrained("facebook/bart-base")
        elif config["iques_architecture"] == "T5":
            self.model = transformers.T5ForConditionalGeneration.from_pretrained("t5-base")
            self.tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-base")
        else:
            raise Exception(
                "Unknown architecture for IQUES module specified in config: currently, only T5-base (=T5) and BART-base (=BART) are supported."
            )

    def set_eval_mode(self):
        """Set model to eval mode."""
        self.model.eval()

    def save(self):
        """Save model."""
        torch.save(self.model.state_dict(), self.iques_model_path)

    def load(self):
        """Load model."""
        if torch.cuda.is_available():
            state_dict = torch.load(self.iques_model_path)
        else:
            state_dict = torch.load(self.iques_model_path, torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def train(self, train_path, dev_path):
        """Train model."""
        # load datasets
        train_dataset = dataset.DatasetSeq2SeqIQUES(self.config, self.tokenizer, train_path)
        dev_dataset = dataset.DatasetSeq2SeqIQUES(self.config, self.tokenizer, dev_path)
        # arguments for training
        training_args = transformers.Seq2SeqTrainingArguments(
            output_dir="faith/temporal_qu/seq2seq_iques/results",  # output directory
            num_train_epochs=self.config["iques_num_train_epochs"],  # total number of training epochs
            per_device_train_batch_size=self.config[
                "iques_per_device_train_batch_size"
            ],  # batch size per device during training
            per_device_eval_batch_size=self.config[
                "iques_per_device_eval_batch_size"
            ],  # batch size for evaluation
            warmup_steps=self.config[
                "iques_warmup_steps"
            ],  # number of warmup steps for learning rate scheduler
            weight_decay=self.config["iques_weight_decay"],  # strength of weight decay
            logging_dir="faith/temporal_qu/seq2seq_iques/logs",  # directory for storing logs
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
            max_length=self.config["iques_max_input_length"],
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            input_encodings = input_encodings.to(torch.device("cuda"))

        # generate
        output = self.model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            no_repeat_ngram_size=self.config["iques_no_repeat_ngram_size"],
            num_beams=self.config["iques_num_beams"],
            early_stopping=self.config["iques_early_stopping"],
            max_length=self.config["iques_max_length"],
        )

        # decoding
        iquess = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        iques = iquess[0]
        return iques

    def inference_top_k(self, input):
        """Run the model on the given input."""
        # encode
        input_encodings = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            max_length=self.config["iques_max_input_length"],
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            input_encodings = input_encodings.to(torch.device("cuda"))
        # generate
        outputs = self.model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            no_repeat_ngram_size=self.config["iques_no_repeat_ngram_size"],
            num_beams=self.config["iques_num_beams"],
            early_stopping=self.config["iques_early_stopping"],
            num_return_sequences=self.config["iques_k"],
            max_length=self.config["iques_max_length"],
        )

        iquess = [
            self.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for output in outputs
        ]
        return iquess

    def inference_on_batch(self, inputs):
        """Run the model on the given inputs (batch)."""
        # encode inputs
        input_encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config["iques_max_input_length"],
            return_tensors="pt",
        )
        # generation
        summary_ids = self.model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            no_repeat_ngram_size=self.config["iques_no_repeat_ngram_size"],
            num_beams=self.config["iques_num_beams"],
            early_stopping=self.config["iques_early_stopping"],
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
