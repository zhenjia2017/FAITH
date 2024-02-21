## _benchmarks/timequestions
This is the benchmark of TimeQuestions.
TimeQuestions contains 16,181 temporal questions.
- Number of data
  - train: 9,708
  - dev: 3,236
  - test: 3,237
  
## _benchmarks/tiq
This is the dataset that combines the Temporal Implicit Questions (TIQ) and the questions with temporal value as answer (Temporal Value Questions, TVQ)
- Number of data
  - train: 13,723 (tiq+tvq)
  - dev: 4,542 (tiq+tvq)
  - test: 4,496 (tiq+tvq)

## We also provide the datasets including TIQ with metadata, TIQ in timequestions format, and Temporal Value Questions respectively.

### _benchmarks/tiq/tiq_with_metadata
This is the benchmark "Temporal Implicit Questions" (TIQ) with metadata including pseudo question etc.
TIQ contains 10,000 implicit temporal questions. The content of each question includes:
- "id"
- "pseudo question"
- "rephrase question"
- "evidence": grounding information snippets
- "signal": OVERLAP, AFTER, BEFORE
- "topic entity": entity for creating the question
- "question entity": entities in the question
- "answer": ground truth answer including Wikidata Qid, Wikidata label, and Wikipedia URL
- "question creation date": question reference time
- "data set": train, dev, or test

### _benchmarks/tiq/tiq_in_timequestions_format
We convert the format of TIQ benchmark into the TimeQuestions format. The content of each question includes:
- "Id"
- "Question"
- "Temporal signal": OVERLAP, AFTER, BEFORE
- "Temporal question type": Implicit
- "Answer": ground truth answer including Wikidata Qid, Wikidata label, and Wikipedia URL
- "Data source": TIQ
- "Question creation date": question reference time
- "Data set": train, dev, or test

### _benchmarks/tiq/temporal_value_questions
We generate intermediate questions using InstructGPT. The explicit temporal value of the implicit constraint
part (from the information snippet in the meta-data of Tiq) is the gold answer to construct <question, date> pairs. If the answer type
of an intermediate question is a time interval, we create two questions asking for "start date" and "end date" respectively, as outlined
before. We constructed 7,723 questions from the train set, 2,542 questions for the dev set, and 2,496 questions for the test set. 

The content of each question includes:
- "Id"
- "Question"
- "Temporal signal": No signal
- "Temporal question type": Temp.Ans
- "Answer": ground truth answer including temporal value and answer type
- "Data source": TIQ-temporal-value-question
- "Question creation date": question reference time
- "Data set": train, dev, or test





