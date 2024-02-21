# Temporal Value Resolver (TVR)

## Description

Module to resolving implicit questions.

- Recursively invoking the temporal QA system itself to obtain the temporal value in TSF
- Give an implicit question, it uses the fine-tuned BART model to generate the intermediate question and its answer type (date or time interval)
- If the answer type is "time interval", the generated questions are rewritten to the two questions with "start date" or "end date" as suffix respectively and answered by the temporal QA system itself
- If the answer type is "date", the generated question is directly answered by the temporal QA system itself
- The top-k (k=1 as default) answer for each question is taken as the temporal value

## Usage
## `resolve_implicit_temporal_value` function
**Inputs**:
- `instance`: an instance in dataset including question

**Outputs**:
- `temporal_values`: temporal values as the slot of TSF
- `iques_answers`: intermediate results for tracing the answer derivation

