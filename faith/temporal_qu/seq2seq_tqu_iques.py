from faith.temporal_qu.temporal_question_understanding import TemporalQuestionUnderstanding
from faith.temporal_qu.seq2seq_tsf.seq2seq_tsf_module import Seq2SeqTSFModule
from faith.temporal_qu.implicit_resolver.seq2seq_iques.seq2seq_iques_module import Seq2SeqIQUESModule
from faith.temporal_qu.implicit_resolver.temporal_value_resolver import TemporalValueResolver
from faith.library.utils import get_logger, get_config


class Seq2SeqIQUESTQU(TemporalQuestionUnderstanding):
    """ Class for our implementation of the TQU. """

    def __init__(self, config, pipeline):
        self.config = config
        self.logger = get_logger(__name__, config)
        # delimiter in temporal structured form for seperating different slots
        self.tsf_delimiter = self.config["tsf_delimiter"]
        # QA system pipeline itself
        self.pipeline = pipeline
        # initiate temporal annotation class
        self.temporal_value_annotator = self.pipeline.fer.temporal_value_annotator
        # question annotation method: SUtime + Regular expression as default
        self.date_tag_method = config["question_date_tag_method"]
        # initiate seq2seq model class for generating TSF
        self.seq2seq_tsf = Seq2SeqTSFModule(config)
        # initiate seq2seq model class for generating intermedaite question in implicit question resolver
        self.seq2seq_iques = Seq2SeqIQUESModule(config)
        # temporal value resolver class for implicit type of questions
        self.run_tvr = self.config["run_tvr"]
        if self.run_tvr:
            # initiate implicit question resolver
            self.tvr = TemporalValueResolver(config, pipeline)
        self.string_lib = self.pipeline.fer.string_lib

    def inference_on_instance(self, instance, topk_answers, sources=["kb", "text", "table", "info"]):
        """
		Implement TQU for the given instance.
		The TSF will be stored in the key `structured_temporal_form`.
		"""
        question = instance["Question"]
        # change the key of "Question" into "question" for consistency
        instance["question"] = question
        # reformat answer and store in `instance["answers"]`
        if "answers" not in instance:
            instance["answers"] = self.string_lib.format_answers(instance)
        question_create_date = instance["Question creation date"]
        # initial TSF is None
        instance["structured_temporal_form"] = None
        # initial temporal resolver is None
        instance["intermediate_question_pipeline_result"] = None

        # Some slots in TSF are generated from seq2seq model, including the question entity, question relation, answer type, temporal signal, and temporal category
        tsf = self.seq2seq_tsf.inference_on_question(question)

        # tsf is a string with the format f"{entities}{tsf_delimiter}{relation}{tsf_delimiter}{ans_type}{tsf_delimiter}{temp_signal}{tsf_delimiter}{temp_category}"
        slots = tsf.split(self.tsf_delimiter)

        qentity = slots[0].strip()
        qrelation = slots[1].strip()
        qanswer_type = slots[2].strip()
        # temporal signal in TimeQuestions: OVERLAP, BEFORE, AFTER, FINISH, START, No signal
        # temporal signal in TIQ: OVERLAP, BEFORE, AFTER
        qtemporal_signal = slots[3].strip()
        # temporal category: implicit; non-implicit
        qtemporal_type = slots[4].strip()

        if "No signal" in qtemporal_signal:
            # remove No signal, in temporal reasoning strategy, take no signal as overlap
            qtemporal_signal = qtemporal_signal.replace("No signal", "")

        # initiate the temporal value slot as a list
        qtemporal_values = list()

        # annotate any ordinals, explicit temporal values
        temporal_annotations = self.temporal_value_annotator.date_ordinal_annotator(question, question_create_date,
                                                                                    self.date_tag_method)
        # record date and ordinal annotation result for each question
        instance["temporal_annotations"] = temporal_annotations
        date_annotations = temporal_annotations[0]
        ordinal_annotations = temporal_annotations[1]
        if date_annotations:
            # data annotation result as constraint
            for date in date_annotations:
                start = date['timespan'][0].replace('T00:00:00Z', '')
                end = date['timespan'][1].replace('T00:00:00Z', '')
                # a timespan is a list with two items of start and end
                qtemporal_values.append([start, end])

        if ordinal_annotations:
            # ordinal annotation result as constraint
            for ordinal in ordinal_annotations:
                # ordinal constraint is a number
                qtemporal_values.append(int(ordinal['ordinal']))

        if "tqu_oracle_temporal_category" in self.config and self.config["tqu_oracle_temporal_category"]:
            # get gold implicit temporal category from meta-data
            if "Temporal question type" in instance:
                if "Implicit" in instance["Temporal question type"]:
                    qtemporal_type = "implicit"

        if "tqu_oracle_temporal_value" in self.config and self.config["tqu_oracle_temporal_value"]:
            # get gold explicit temporal value from meta-data
            qtemporal_values = []
            if "gold_constraint" in instance:
                for item in instance["gold_constraint"]:
                    qtemporal_values.append(item)

            tsf = self._output_tsf(qtemporal_type, qentity, qrelation, qanswer_type, qtemporal_signal, qtemporal_values)

            instance["structured_temporal_form"] = tsf

            return instance

        # translate implicit constraint into temporal value when there is no other constraint
        if qtemporal_type == "implicit" and self.run_tvr:
            # implicit resolver
            temporal_value, iques_instance = self.tvr.resolve_implicit_temporal_value(instance, topk_answers, sources)
            if temporal_value:
                qtemporal_values += temporal_value
                if iques_instance:
                    instance["intermediate_question_pipeline_result"] = iques_instance
            else:
                self.logger.info(
                    f"Fail to generate intermediate question or get temporal value for question {question}")

        tsf = self._output_tsf(qtemporal_type, qentity, qrelation, qanswer_type, qtemporal_signal, qtemporal_values)

        instance["structured_temporal_form"] = tsf

        return instance

    def _output_tsf(self, qtemporal_type, entity, relation, answer_type, temporal_signal, temporal_values):
        tsf = {
            "entity": entity,  # string
            "relation": relation,  # relation is a string
            "answer_type": answer_type,  # string
            "temporal_signal": temporal_signal,  # temporal signal is a string
            "category": qtemporal_type,  # non-implicit, implicit
            "temporal_value": temporal_values  # temporal value is a list
        }
        return tsf

    def train(self):
        """ Abstract training function that triggers training of submodules. """
        self.seq2seq_tsf.train()
        self.seq2seq_iques.train()


#######################################################################################################################
#######################################################################################################################
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception(
            "Usage: python faith/temporal_qu/temporal_question_understanding.py --<FUNCTION> <PATH_TO_CONFIG>"
        )

    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    # train: train model
    if function == "--train":
        tqu = TemporalQuestionUnderstanding(config)
        tqu.train()

    # inference: add predictions to data
    elif function == "--inference":
        # load config
        topk_answers = 1
        tqu = TemporalQuestionUnderstanding(config)
        tqu.inference(topk_answers)

    # inference: add predictions to data
    elif function == "--example":
        # load config
        instance = ''
        topk_answers = 1
        tqu = TemporalQuestionUnderstanding(config)
        res = tqu.inference_on_instance(instance, topk_answers)
        print(res)
