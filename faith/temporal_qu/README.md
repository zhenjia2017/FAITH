# Temporal Question Understanding (TQU)

Module to create TSF for the questions.

- [Create your own QU module](#create-your-own-tqu-module)
  - [`inference_on_instance` function](#inference_on_instance-function)
  - [`train` function](#optional-train-function)

## Create your own TQU module
You can inherit from the [TemporalQuestionUnderstanding](temporal_question_understanding.py) class and create your own TQU module. Implementing the functions `inference_on_instance` is sufficient for the pipeline to run properly. You might want to implement your own training procedure for your module via the `train` function though.

Further, you need to instantiate a logger in the class, which will be used in the parent class.
Alternatively, you can call the __init__ method of the parent class.

## `inference_on_instance` function

**Inputs**:
- `instance`: the current question for which the TSF should be generated.

**Description**:  
This method is supposed to generate the TSF for the current question.

**Output**:  
Returns the instance. Make sure to store the TSF of the information need in `instance["structured_temporal_form"]`. 

## `inference_on_data_benchmark_split` function
**Inputs**:
- `benchmark`: the benchmark name of questions for which the TSF should be generated, e.g., TIQ or TimeQuestions.
- `split`: the dataset split of questions such as "train", "dev" or "test".

**Description**:  
This method is supposed to generate the TSF for all questions in a dataset split of a benchmark.

**Output**:  
Returns all the instances in a dataset split and stores the results. 

## `evaluate_tqu` function
**Inputs**:
- `dataset file path`: dataset file path in which all questions are with generated TSF.

**Description**:  
This method is supposed to evaluate the accuracy of temporal signal and temporal category in TSF.

**Output**:  
Stores the evaluation results. 


## [Optional] `train` function

**Inputs**: NONE

**Description**:  
If required, you can train your TQU module here. You can make use of whatever parameters are stored in your .yml file.

**Output**: NONE
