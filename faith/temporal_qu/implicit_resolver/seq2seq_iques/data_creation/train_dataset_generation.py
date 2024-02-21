import openai
import json
import os
import time
import pickle
from filelock import FileLock
from pathlib import Path
from tqdm import tqdm

from faith.library.utils import get_config, get_logger

"""
The script is used to generate train and dev data set for fine-tuning the BART model
"""

# static variable: prompt
#this prompt is for generating intermediate questions for TimeQuestions
GENERATE_PROMPT_TIMESPAN_TIMEQUESTIONS = """
            Generate an explicit question and answer type for the implicit part of the temporal input question.\n

            Input: what position did djuanda kartawidjaja take after he was replaced by sukarano
            Output: when djuanda kartawidjaja replaced by sukarano||date

            Input: american naval leader during the world war 2
            Output: when world war 2||time interval

            Input: who became president after harding died
            Output: when harding died||date

            Input: who did luis suarez play for before liverpool
            Output: when luis suarez play for liverpool||time interval

            Input: which countries were located within the soviet union prior to its dissolution
            Output: when soviet union dissolution||date

            Input: who started the presidency earliest and served as president during wwii in the US
            Output: when wwii||time interval

            Input: who replaced aldo moro as the minister of foreign affairs
            Output: when aldo moro replaced as minister of foreign affairs||date

            Input: what did harry s truman work before he was president
            Output: when harry s truman president||time interval

            Input: """

# static variable: prompt
#this prompt is for generating intermediate questions for TIQ
GENERATE_PROMPT_TIMESPAN_TIQ = """
            Generate an explicit question and answer type for the implicit part of the temporal input question.\n

            Input: Who was the second director of the Isabella Stewart Gardner Museum when it was built
            Output: When Isabella Stewart Gardner Museum was built||time interval

            Input: When Wendy Doniger was president of the Association for Asian Studies, what publishing house was she based in New York
            Output: When Wendy Doniger was president of the Association for Asian Studies||time interval

            Input: What administrative entity was Ezhou in before Huangzhou District became part of it
            Output: When Huangzhou District became part of Ezhou||date

            Input: After Bud Yorkin became the producer of NBC's The Tony Martin Show, who was his spouse?
            Output: When Bud Yorkin became the producer of NBC's The Tony Martin Show||date

            Input: What book did Ira Levin write that was adapted into a film during the same time he wrote the play Deathtrap
            Output: When Ira Levin wrote the play Deathtrap||date

            Input: What basketball team was Nathaniel Clifton playing for when his career history with the Rens began
            Output: When Nathaniel Clifton's career history with the Rens began||time interval

            Input: What team did Stevica Ristić play for before joining Shonan Bellmare?
            Output: When Stevica Ristić joining Shonan Bellmare||time interval

            Input: Which album was released by the Smashing Pumpkins after Mike Byrne joined the band
            Output: When Mike Byrne joined Smashing Pumpkins||time interval

            Input: """


class TrainDataset():
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.data_path = self.config["path_to_data"]
        self.benchmark = self.config["benchmark"]
        self.model = self.config["gpt3_model_iques"]
        self.use_cache = self.config["gpt3_use_cache_iques"]

        # initialize gpt output cache dictionary
        if self.use_cache:
            self._init_cache()
            self.cache_changed = False

    def _prompt_chat_gpt(self, question_prompt):
        ## WITH CHAT GPT
        ## WITH INSTRUCT GPT
        openai.organization = self.config["openai_organization"]
        openai.api_key = self.config["openai_api_key"]
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": question_prompt}]
            )
            generated_question = response.choices[0].message.content.strip()
            self.cache[question_prompt] = generated_question
            self.cache_changed = True
        except:
            print(f"FAIL: GPT did not respond for the following question_prompt: {question_prompt}")
            # print(f"Response: {response}")
            generated_question = "None"
        return generated_question

    def _prompt_instruct_gpt(self, question_prompt):
        ## WITH INSTRUCT GPT
        openai.organization = self.config["openai_organization"]
        openai.api_key = self.config["openai_api_key"]
        try:
            response = openai.Completion.create(
                model=self.model,
                prompt=question_prompt,
                temperature=1.0,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            generated_question = response["choices"][0]["text"].strip()
            self.cache[question_prompt] = generated_question
            self.cache_changed = True
            print(f"GPT generate question: {generated_question}")
        except:
            print(f"FAIL: GPT did not respond for the following question_prompt: {question_prompt}")
            generated_question = "None"
        return generated_question

    def gpt_generate_question(self, question_prompt):

        if question_prompt in self.cache:
            print(f"Get generate question from cache")
            return self.cache[question_prompt]

        # generate output
        if self.model == "gpt-3.5-turbo":
            return self._prompt_chat_gpt(question_prompt)
        elif self.model == "text-davinci-003":
            return self._prompt_instruct_gpt(question_prompt)

    def generate_intermediate_question(self, prompt, instance):
        """
        Generate intermediate question for the questions with implicit type.
        """
        input_question = instance["Question"]
        print("Question:", input_question)
        question_prompt = prompt + input_question + "\nOutput:"

        return self.gpt_generate_question(question_prompt)


    def process_on_data_split(self, input_path, output_path, benchmark, split="train"):
        """
        Run question generation on the dataset to get training data
        """
        # open data
        with open(input_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        self.logger.info(f"Input data loaded from: {input_path}.")

        # create folder if not exists
        Path(output_path).mkdir(parents=True, exist_ok=True)

        instances = []
        # process data
        for instance in tqdm(data):
            if benchmark == "timequestions" and "Implicit" not in instance["Temporal question type"]:
                continue
            elif benchmark == "timequestions" and "Implicit" in instance["Temporal question type"]:
                prompt = GENERATE_PROMPT_TIMESPAN_TIMEQUESTIONS
            elif benchmark == "timequestions_alternative_answer" and "Implicit" not in instance["Temporal question type"]:
                continue
            elif benchmark == "timequestions_alternative_answer" and "Implicit" in instance["Temporal question type"]:
                prompt = GENERATE_PROMPT_TIMESPAN_TIMEQUESTIONS
            elif benchmark == "tiq_with_metadata":
                prompt = GENERATE_PROMPT_TIMESPAN_TIQ

            generated_question = self.generate_intermediate_question(prompt, instance)
            # only the generated content containing "||" are remained for further processing
            if "||" in generated_question:
                intermediate_question = generated_question.split("||")[0]
                type = generated_question.split("||")[1]
                #only the generated type is date or time interval is saved
                if not (type == "date" or type == "time interval"):
                    continue
                instance["silver_generated_question"] = [intermediate_question, type]
                instances.append(instance)

        with open(os.path.join(output_path, f'gpt_annotate_generate_question_{split}.json'), "w", encoding="utf-8") as fp:
            fp.write(json.dumps(instances, ensure_ascii=False, indent=4))

    def store_cache(self):
        """Store the cache to disk."""
        if not self.use_cache:  # store only if cache in use
            return
        if not self.cache_changed:  # store only if cache changed
            return
        # check if the cache was updated by other processes
        if self._read_cache_version() == self.cache_version:
            # no updates: store and update version
            self.logger.info(f"Writing GPT3 cache at path {self.cache_path}.")
            with FileLock(f"{self.cache_path}.lock"):
                self._write_cache(self.cache)
                self._write_cache_version()
        else:
            # update! read updated version and merge the caches
            self.logger.info(f"Merging GPT3 cache at path {self.cache_path}.")
            with FileLock(f"{self.cache_path}.lock"):
                # read updated version
                updated_cache = self._read_cache()
                # overwrite with changes in current process (most recent)
                updated_cache.update(self.cache)
                # store
                self._write_cache(updated_cache)
                self._write_cache_version()

    def _init_cache(self):
        """Initialize the cache."""
        self.cache_dir = self.config["gpt3_cache_path_iques"]
        self.cache_dir = os.path.join(self.data_path, self.benchmark, self.cache_dir)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, "gpt_cache_intermediate_question.pickle")
        if os.path.isfile(self.cache_path):
            # remember version read initially
            self.logger.info(f"Loading GPT3 cache from path {self.cache_path}.")
            with FileLock(f"{self.cache_path}.lock"):
                self.cache_version = self._read_cache_version()
                self.logger.debug(self.cache_version)
                self.cache = self._read_cache()
            self.logger.info(f"GPT3 cache successfully loaded.")
        else:
            self.logger.info(f"Could not find an existing GPT3 cache at path {self.cache_path}.")
            self.logger.info("Populating GPT3 cache from scratch!")
            self.cache = {}
            self._write_cache(self.cache)
            self._write_cache_version()

    def _read_cache(self):
        """
        Read the current version of the cache.
        This can be different from the version used in this file,
        given that multiple processes may access it simultaneously.
        """
        # read file content from cache shared across QU methods
        with open(self.cache_path, "rb") as fp:
            cache = pickle.load(fp)
        return cache

    def _write_cache(self, cache):
        """Write to the cache."""
        cache_dir = os.path.dirname(self.cache_path)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as fp:
            pickle.dump(cache, fp)
        return cache

    def _read_cache_version(self):
        """Read the cache version (hashed timestamp of last update) from a dedicated file."""
        if not os.path.isfile(f"{self.cache_path}.version"):
            self._write_cache_version()
        with open(f"{self.cache_path}.version", "r") as fp:
            cache_version = fp.readline().strip()
        return cache_version

    def _write_cache_version(self):
        """Write the current cache version (hashed timestamp of current update)."""
        with open(f"{self.cache_path}.version", "w") as fp:
            version = str(time.time())
            fp.write(version)
        self.cache_version = version

#######################################################################################################################
#######################################################################################################################
import sys

if __name__ == "__main__":
    # RUN: python script config file
    if len(sys.argv) < 3:
        raise Exception(
            "Usage: python faith/temporal_qu/dataset_creation/train_dataset_generation.py <FUNCTION> <PATH_TO_CONFIG>"
        )

    # load config
    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)
    # load config
    generator = TrainDataset(config)
    benchmark = config["benchmark"]
    output_path = os.path.join(config["path_to_data"], benchmark, config["intermediate_question_dataset_path"])

    if function == "--train":
        input_file = os.path.join(config["benchmark_path"], benchmark, config["train_input_path"])
        generator.process_on_data_split(input_file, output_path, benchmark, split="train")

    elif function == "--dev":
        input_file = os.path.join(config["benchmark_path"], benchmark, config["dev_input_path"])
        generator.process_on_data_split(input_file, output_path, benchmark, split="dev")

    generator.store_cache()




