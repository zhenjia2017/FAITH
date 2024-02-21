# Distant Supervision

- [Usage](#usage)
- [Input format](#input-format)
- [Output format](#output-format)

## Usage
For running the distant supervision on a given dataset, simply run:
```
python faith/distant_supervision/distant_supervision.py --function <PATH_TO_CONFIG>
```
or run the script:
``` bash
bash scripts/distant_supervision.sh <PATH_TO_CONFIG> <PATH_TO_CONFIG>
```

from the ROOT directory of the project.  
The paths to the input files will be read from the given config values for `train_input_path`, `dev_input_path`, and `test_input_path`.
This will create annotated versions of the benchmark in `_data/tsf_annotation/<BENCHMARK>/`.

## Input format
The annotation script expects the benchmark in the following (minimal) format:
```
[
	// first question
	{	
		"Question": "<QUESTION>",
		"Answer": [{
			"AnswerType": "Entity",
			"WikidataQid": "Q5",
			"WikidataLabel": "human"
		},{
			"AnswerType": "Value",
			"AnswerArgument": "2022-09-29T00:00:00Z",
		}]
	},
	// second question
	{
		...
	},
	// ...
]
```
Any other keys can be provided, and will be written to the output.
You can see [here](../heterogeneous_answering#answer-format) for additional information of the expected format of the answer IDs and labels.

## Output format
The result will be stored in a .json file:

```
[
	// first question
	{	
		"question": "<QUESTION>", 
		"answers": [{
			"id": "<Wikidata ID>",
			"label": "<Item Label>
			},
			//...
		]
		// inferred TSF
		"silver_tsf": 
		["<TSF1>", "<TSF2>",]
	},
	// second question
	{
		...
	},
	// ...
]
```
