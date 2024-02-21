# Intermediate Question Dataset Creation

## Description
Module to create intermediate questions and their answer type (date or time interval) for fine-tuning the BART model.

We use in-context learning to obtain the data set: 
 - Select and label 8 questions as input. The output is the intermediate question that describes the implicit constraint, and the expected
answer type for this question, separated by two pipes (“||”): “{intermediate question}||{expected answer type}”
 - Give these pairs as context to a LLM (InstructGPT), to annotate the remaining questions in the train and dev set.
 - The amount of dataset
   - For TimeQuestions, we generate 847 train data (from 883 questions in train) and 287 dev data (from 296 questions in dev). 
   - For TIQ, we generate 5,875 train data (from 6000 questions in train) and 1,949 dev data (from 2000 questions in dev).
   - Due to the errors such as the answer type is not "date" or "time interval", we drop the output of the LLM and the number of generated intermediate questions is not the same as the number of implicit questions in the dataset.  

## Usage
- You need to set the openai_api_key and openai_organization code in the configuration file
- The model we use is text-davinci-003
- You need to set the data path of TimeQuestions and TIQ
- If you want to cache the result from LLM, you need to set the cache file path in the configuration file
- You can run the script as follows to reproduce the result
    python train_dataset_generation.py \<FUNCTION\> \<PATH_TO_CONFIG\> to reproduce the results

## `process_on_data_split` function

**Inputs**:
- `input_file`: dataset file path
- `output_path`: output folder
- `benchmark`: benchmark name
- `split`: "train", "dev" or "test"

**Outputs**:
- `train`: gpt_annotate_generate_question_train.json
- `dev`: gpt_annotate_generate_question_dev.json

## Usage
For running the script on a given dataset, simply run:
```
python faith/temporal_qu/implicit_resolver/seq2seq_iques/data_creation/train_dataset_generation.py --function <PATH_TO_CONFIG>
```
or run the script:
``` bash
bash scripts/intermediate_question_generation.sh <PATH_TO_CONFIG> <PATH_TO_CONFIG>
```

**Prompts**:

The prompts ultimately used for creating the training dataset for the generation of intermediate questions for TIQ.
            
            Generate an explicit question and answer type for the implicit part of the temporal input question.

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

            Input:

The prompts ultimately used for creating the training dataset for the generation of intermediate questions for TimeQuestions.

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

            Input:
