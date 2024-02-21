from faith.library.utils import get_logger


class EvidencePruning:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)

    def prune_on_instance(self, instance, sources):
        tsf = instance["structured_temporal_form"]
        evidences = instance["candidate_evidences"]
        faithful_evidences = self.pruning_evidences(tsf, evidences, sources)
        return faithful_evidences

    def pruning_evidences(self, tsf, evidences, sources):
        temporal_signal = tsf["temporal_signal"]
        constraint = tsf["temporal_value"]
        answer_type = tsf["answer_type"]
        faithful_evidences = []

        if 'date' in answer_type or "year" in answer_type:
            # if questions asking for a date or a year, the evidence without any temporal information is pruned out
            for evidence in evidences:
                if evidence["source"] not in sources: continue
                if not evidence["tempinfo"]: continue
                faithful_evidences.append(evidence)
                self.logger.debug(f"Found evidence for this temporal value TSF: {tsf}.")

        elif constraint:
            question_constraints = list()
            for item in constraint:
                if isinstance(item, int):
                    # if the item is an integar, it is an ordinal number
                    continue
                elif isinstance(item, list):
                    # if the item is a list, it is a temporal interval
                    question_constraints.append(item)
            # normalize the constraint of temporal intervals into valid timespans
            normalized_question_timespans = self.normalize_timespan(question_constraints)
            if normalized_question_timespans:
                # constraint is timespan
                for evidence in evidences:
                    if evidence["source"] not in sources: continue
                    if not evidence["tempinfo"]: continue
                    timespan, disambiguate = evidence["tempinfo"]
                    evidence_timespans = self.normalize_timespan(timespan)
                    if len(evidence_timespans) == 0: continue
                    if self.reasonbytimespan(temporal_signal, evidence_timespans, normalized_question_timespans):
                        # self.logger.debug(f"Found faithful evidence for this Explicit TSF: {tsf}.")
                        # print (f"{evidence_timespans} satisfy the signal {temporal_signal} with constraint {question_timespans}")
                        faithful_evidences.append(evidence)
            else:
                # constraint is ordinal number
                for evidence in evidences:
                    if evidence["source"] not in sources: continue
                    faithful_evidences.append(evidence)
        else:
            # when answer type is not a date and the temporal value is null, we keep all evidences as faithful evidence
            for evidence in evidences:
                if evidence["source"] not in sources: continue
                faithful_evidences.append(evidence)

        # for other types of questions such as ordinal, we solve these questions in the future
        return faithful_evidences

    def reasonbytimespan(self, signal, evidence_timespans, question_timespans):
        if signal == 'BEFORE':
            return self.reason_before(evidence_timespans, question_timespans)
        elif signal == 'AFTER':
            return self.reason_after(evidence_timespans, question_timespans)
        elif signal == 'START':
            return self.reason_start(evidence_timespans, question_timespans)
        elif signal == 'FINISH':
            return self.reason_finish(evidence_timespans, question_timespans)
        else:
            return self.reason_overlap(evidence_timespans, question_timespans)

    def reason_overlap(self, main_timespans, constraint_timespans):
        for main_timespan in main_timespans:
            for constraint_timespan in constraint_timespans:
                evi_begin = main_timespan[0]
                evi_end = main_timespan[1]
                constraint_start = constraint_timespan[0]
                constraint_end = constraint_timespan[1]
                if "0101" in str(evi_begin)[-4:] and "1231" in str(evi_end)[-4:] and "0101" in str(constraint_start)[
                                                                                               -4:] and "1231" in str(
                        constraint_end)[-4:]:
                    # evi_end is a year and constraint_start is a year
                    evi_end = int(str(evi_end)[:-4] + "0101")
                    constraint_end = int(str(constraint_end)[:-4] + "0101")

                if main_timespan == constraint_timespan:
                    return True

                if evi_begin <= constraint_start and evi_end >= constraint_end:
                    # evidence timespan contains constraint timespan
                    return True
                if evi_begin >= constraint_start and evi_end <= constraint_end:
                    # constraint timespan contains evidence timespan
                    return True
                if evi_begin <= constraint_start and evi_end >= constraint_start and evi_end <= constraint_end:
                    return True
                if evi_end >= constraint_end and evi_begin >= constraint_start and evi_begin <= constraint_end:
                    return True

        return False

    def reason_after(self, main_timespans, constraint_timespans):
        for main_timespan in main_timespans:
            evi_begin = main_timespan[0]
            for constraint_timespan in constraint_timespans:
                constraint_start = constraint_timespan[0]
                constraint_end = constraint_timespan[1]
                if "0101" in str(constraint_start)[-4:] and "1231" in str(constraint_end)[-4:] and "0101" in str(
                        evi_begin)[4:]:
                    # constraint a year
                    constraint_end = int(str(constraint_end)[:-4] + "0101")
                if evi_begin >= constraint_end:
                    return True
        return False

    def reason_before(self, main_timespans, constraint_timespans):
        for main_timespan in main_timespans:
            evi_begin = main_timespan[0]
            evi_end = main_timespan[1]
            for constraint_timespan in constraint_timespans:
                constraint_start = constraint_timespan[0]
                if "0101" in str(evi_begin)[-4:] and "1231" in str(evi_end)[-4:] and "0101" in str(constraint_start)[
                                                                                               -4:]:
                    # evi_end is a year and constraint_start is a year
                    evi_end = int(str(evi_end)[:-4] + "0101")

                if evi_end <= constraint_start:
                    return True
        return False

    def reason_finish(self, main_timespans, constraint_timespans):
        for main_timespan in main_timespans:
            evi_end = main_timespan[1]
            for constraint_timespan in constraint_timespans:
                constraint_end = constraint_timespan[1]
                if evi_end == constraint_end:
                    return True
        return False

    def reason_start(self, main_timespans, constraint_timespans):
        for main_timespan in main_timespans:
            evi_begin = main_timespan[0]
            for constraint_timespan in constraint_timespans:
                constraint_start = constraint_timespan[0]
                if evi_begin == constraint_start:
                    return True
        return False

    def normalize_timespan(self, timespans):
        normalized_timespans = []
        if len(timespans) == 0:
            return normalized_timespans
        for timespan in timespans:
            start_time_int = float('-inf')
            end_time_int = float('inf')
            if timespan[0]:
                if "T00:00:00Z" in timespan[0]:
                    time_str = timespan[0].replace("T00:00:00Z", "")
                else:
                    time_str = timespan[0]
                start_time = time_str.strip()
                if start_time.startswith("-"):
                    try:
                        start_time_int = int(f"-{start_time.replace('-', '')}")
                    except:
                        print(f"Fail to convert the date into integer {start_time}")
                else:
                    try:
                        start_time_int = int(f"{start_time.replace('-', '')}")
                    except:
                        print(f"Fail to convert the date into integer {start_time}")
            if timespan[1]:
                if "T00:00:00Z" in timespan[1]:
                    time_str = timespan[1].replace("T00:00:00Z", "")
                else:
                    time_str = timespan[1]
                end_time = time_str.strip()
                if end_time.startswith("-"):
                    try:
                        end_time_int = int(f"-{end_time.replace('-', '')}")
                    except:
                        print(f"Fail to convert the date into integer {end_time}")
                else:
                    try:
                        end_time_int = int(f"{end_time.replace('-', '')}")
                    except:
                        print(f"Fail to convert the date into integer {end_time}")

            if start_time_int > end_time_int:
                # ignore the timespans which are not accurate
                continue
            normalized_timespans.append([start_time_int, end_time_int])

        return normalized_timespans
