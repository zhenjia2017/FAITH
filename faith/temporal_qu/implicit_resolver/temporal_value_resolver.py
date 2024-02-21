from faith.library.utils import get_logger
from faith.temporal_qu.implicit_resolver.seq2seq_iques.seq2seq_iques_module import Seq2SeqIQUESModule


class TemporalValueResolver:
    def __init__(self, config, pipeline):
        self.config = config
        self.logger = get_logger(__name__, config)
        # instantiate fine-tuned BART (Seq2Seq) model to generate intermediate question
        self.iques_generation = Seq2SeqIQUESModule(config)
        self.delimiter = self.config["tsf_delimiter"]
        self.name = config["name"]
        # temporal QA system pipeline for iteratively answering the intermediate question
        self.pipeline = pipeline
        # library for annotating date based on regular expression
        self.string_lib = self.pipeline.fer.string_lib

    def resolve_implicit_temporal_value(self, instance, topk_answers, sources=["kb", "text", "table", "info"]):
        iques_answers = []
        temporal_values = []
        # inference_on_instance function generates intermediate question using the fine-tuned BART model
        # inference_on_instance function takes an instance which is a dictionary as input and output the intermediate question with its answer type saved in the instance
        self.iques_generation.inference_on_instance(instance)
        result = instance["generated_iquestion"]
        self.logger.info(f"Generate question: {result}")

        if self.delimiter in result:
            generated_q = result.split(self.delimiter)[0].strip()
            answer_type = result.split(self.delimiter)[1].strip()
            # if the generated question is same as original question, the system will be in endless loop to call system itself
            if generated_q.lower() != instance["Question"].lower():
                # generate timespans
                if answer_type == "time interval":
                    suffix = ["start date", "end date"]
                    timestamps = {suffix[0]: [], suffix[1]: []}
                    for item in suffix:
                        # construct intermediate question with start date or end date as suffix
                        question_suffix = f"{generated_q} {item}"
                        iques_instance = {}
                        iques_instance.update({"answers": instance["answers"]})
                        iques_instance.update({"Id": instance["Id"]})
                        iques_instance.update({"Question creation date": instance["Question creation date"]})
                        iques_instance.update(
                            {"generated_q": generated_q, "question": question_suffix, "Question": question_suffix,
                             "intermediate_q_answer_type": answer_type})
                        # call the QA pipeline to answer the intermediate question
                        iques_instance = self.pipeline.inference_on_instance(iques_instance, topk_answers, sources)
                        ranked_answers = iques_instance["ranked_answers"]
                        # obtain the top-k answer of temporal value
                        temporal_value = self.extract_temporal_value(topk_answers, ranked_answers)  # extract temporal value from answer
                        # remember results
                        iques_instance["temporal_value"] = temporal_value
                        iques_answers.append(iques_instance)
                        if temporal_value:
                            timestamps[item] += temporal_value
                    # generate exhaustive set of possible timespan(s) from timestamp candidates
                    temporal_values += self._generate_timespans(timestamps)
                    self.logger.info(f"temporal value: {temporal_values}")

                else:
                    iques_instance = {}
                    iques_instance.update({"answers": instance["answers"]})
                    iques_instance.update({"Id": instance["Id"]})
                    iques_instance.update({"Question creation date": instance["Question creation date"]})
                    iques_instance.update({"generated_q": generated_q, "question": generated_q, "Question": generated_q,
                                           "iques_answer_type": answer_type})
                    # call the QA pipeline to answer the intermediate question
                    iques_instance = self.pipeline.inference_on_instance(iques_instance, topk_answers, sources)
                    ranked_answers = iques_instance["ranked_answers"]
                    # call the QA pipeline to answer the intermediate question
                    temporal_value = self.extract_temporal_value(topk_answers, ranked_answers)  # extract temporal value from answer
                    iques_instance["temporal_value"] = temporal_value
                    iques_answers.append(iques_instance)
                    if temporal_value:
                        for timestamp in temporal_value:
                            if "-01-01" in timestamp:
                                # when the timestamp is a year, we extend the timestamp as a time span for a year
                                timespan = [timestamp, timestamp.replace("-01-01", "-12-31")]
                                temporal_values.append(timespan)
                            else:
                                # when the timestamp is a date, we keep the timestamp as the start date and end date of a timespan
                                timespan = [timestamp, timestamp]
                                temporal_values.append(timespan)
                        self.logger.info(f"temporal value: {temporal_values}")

        return temporal_values, iques_answers

    def extract_temporal_value(self, topk_answers, ranked_answers):
        timestamps_for_implicitquestion = list()
        pred_answers = [{"id": ans["answer"]["id"], "label": ans["answer"]["label"], "rank": ans["rank"]} for ans in
                        ranked_answers]
        if pred_answers:
            for answer in pred_answers:
                if answer["rank"] > topk_answers:
                    self.logger.info(
                        f"Topk date answers for implicit question: {timestamps_for_implicitquestion}.")
                    break
                if self.string_lib.is_timestamp(answer["id"]):
                    timestamps_for_implicitquestion.append(answer["id"].replace('T00:00:00Z', ''))

        return timestamps_for_implicitquestion

    def _generate_timespans(self, timestamps):
        """
        Generate exhaustive set of possible timespan(s) from timestamp candidates.
        Checks if start time and end time match in temporal scope.
        timestamps: [answer for start time, answer for end time]
        """
        timespans = []
        start_dates = timestamps["start date"]
        end_dates = timestamps["end date"]
        if len(start_dates) > 0 and len(end_dates) > 0:
            for start in start_dates:
                for end in end_dates:
                    # start date should be less than end date otherwise we drop the date
                    if start <= end:
                        if start == end and "-01-01" in end:
                            # if the start and end is a year, extend it to a year period
                            end = end.replace("-01-01", "-12-31")
                        timespans.append([start, end])
            # if the start date and end date can't group into a timespan, we keep them respectively
            if len(timespans) == 0:
                for start in start_dates:
                    timespans.append([start, None])
                for end in end_dates:
                    timespans.append([None, end])
        # if there is no end date as answer, we keep start date in the timespan
        elif len(start_dates) > 0 and len(end_dates) == 0:
            for start in start_dates:
                timespans.append([start, None])
        # if there is no start date as answer, we keep end date in the timespan
        elif len(start_dates) == 0 and len(end_dates) > 0:
            for end in end_dates:
                timespans.append([None, end])

        return timespans
