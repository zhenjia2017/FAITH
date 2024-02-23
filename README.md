# FAITH

Description
------
This repository contains the code, benchmark and data for our WWW'24 full paper "Faithful Temporal Question Answering over Heterogeneous Sources". In this paper, we present FAITH, 
for faithfully answering temporal questions with tangible evidence. 

<div style="text-align: center;"><img src="faith-overview-figure.png"  alt="overview" width=80%  /></div>

*Overview of the FAITH pipeline. The figure illustrates the process for answering q<sub>3</sub> (“Queen’s record company when recording
Bohemian Rhapsody?” ) and q<sub>1</sub> (“Record company of Queen in 1975?” ). For answering q<sub>3</sub>, two intermediate questions q<sub>31</sub> and q<sub>32</sub>
are generated, and run recursively through the entire temporal QA system.*

For more details see our paper: [Faithful Temporal Question Answering over Heterogeneous Sources]() and visit our project website: https://faith.mpi-inf.mpg.de.

If you use this code, please cite:
```bibtex
@article{jia2024faithful,
  title={Faithful Temporal Question Answering over Heterogeneous Sources},
  author={Jia, Zhen and Christmann, Philipp and Weikum, Gerhard},
  journal={WWW},
  year={2024}
```

## Environment setup
We recommend the installation via conda, and provide the corresponding environment file in [environment.yml](environment.yml):

```bash
  git clone https://github.com/zhenjia2017/FAITH.git
  cd FAITH/
  conda env create --file environment.yml
  conda activate faith
  pip install -e .
```
Alternatively, you can also install the requirements via pip, using the [requirements.txt](requirements.txt) file. 

### Dependencies
FAITH makes use of [CLOCQ](https://github.com/PhilippChr/CLOCQ) for retrieving facts from WIKIDATA.
CLOCQ can be conveniently integrated via the [publicly available API](https://clocq.mpi-inf.mpg.de), using the client from [the repo](https://github.com/PhilippChr/CLOCQ).  

FAITH makes use of [SUTIME](https://nlp.stanford.edu/software/sutime.html) for annotating explicit dates in questions. 
You can install [python-sutime](https://github.com/FraBle/python-sutime) via the following command.
```bash
  pip install sutime
  mvn dependency:copy-dependencies -DoutputDirectory=./jars -f $(python3 -c 'from importlib import util; import pathlib; print(pathlib.Path(util.find_spec("sutime").origin).parent / "pom.xml")')
```
Then you can run the script for starting [SUTIME](https://nlp.stanford.edu/software/sutime.html) as a backend service. 
```bash
  bash scripts/start_sutime_server.sh
```

### Data
You need the following data:
- wikipedia_wikidata_mappings.pickle
- wikipedia_mappings.pickle
- wikidata_mappings.pickle
- types.pickle
- stopwords.txt
- labels.pickle
- augmented_wikidata_mappings.pickle

We provide the trained model and data for reproducing the results. You can download from [here](https://qa.mpi-inf.mpg.de/faith/data_for_reproduce_faith.tar.gz) 
(unzip and put it in the "_data" folder; total data size around 20 GB).
The data folder structure is as follows:

```
_data
├──tiq (or timequestions)
    ├── intermediate_question
    ├── tsf_annotation
    ├── iques_model.bin
    ├── tsf_model.bin
    ├── faith
        ├──sbert_model.bin
        ├──seq2seq_ha
        └──explaignn
            ├── gnn-answering-ignn-100-05-05.bin
            └── gnn-pruning-ignn-100-00-10.bin
    ├── unfaith
        ├──sbert_model.bin
        └──explaignn
            ├── gnn-answering-ignn-100-05-05.bin
            └── gnn-pruning-ignn-100-00-10.bin
├── wikipedia_wikidata_mappings.pickle    
├── wikipedia_mappings.pickle
├── wikidata_mappings.pickle
├── types.pickle
├── stopwords.txt
├── labels.pickle
└── augmented_wikidata_mappings.pickle
```

- tiq/intermediate_question: generated intermediate questions from GPT as training dataset for fine-tuning BART model on TIQ benchmark
- tiq/tsf_annotation: TSF training data generated via distant supervision for fine-tuning BART model on TIQ benchmark
- tiq/iques_model.bin: fine-tuned BART model for generating intermediate questions in TQU stage
- tiq/tsf_model.bin: fine-tuned BART model for generating TSF in TQU stage
- tiq/faith/sbert_model.bin: fine-tuned BERT model for scoring evidence in FER stage
- tiq/faith/seq2seq_ha: fine-tuned BART model for heterogeneous answering in HA stage
- tiq/faith/explaignn/gnn-pruning-ignn-100-00-10.bin: graph reduction model in HA stage
- tiq/faith/explaignn/gnn-answering-ignn-100-05-05.bin: answer inference model in HA stage
- tiq/unfaith/sbert_model.bin: fine-tuned BERT model for scoring evidence in FER stage for UNFAITH settings
- tiq/unfaith/explaignn/gnn-pruning-ignn-100-00-10.bin: graph reduction model in HA stage for UNFAITH settings
- tiq/unfaith/explaignn/gnn-answering-ignn-100-05-05.bin: answer inference model in HA stage for UNFAITH settings

## Reproduce paper results
### Main results
Please run the following script to reproduce the main results of FAITH (Table 3 in the WWW 2024 paper):
```bash
  bash scripts/pipeline.sh --evaluate <PATH_TO_CONFIG>
```
For example,
```bash
  bash scripts/pipeline.sh --evaluate config/evaluate.yml
```
If you want to reproduce the results of UNFAITH, please set **faith_or_unfaith** as **unfaith** in the config file.

### Training
    
There are three stages in FAITH: **T**emporal **Q**uestion **U**nderstanding (**TQU**), **F**aithful **E**vidence **R**etrieval (**FER**), and **E**xplainable **H**eterogeneous **A**nswering (**EHA**).

#### TQU
In the TQU stage, there are two Seq2seq models for:
  - (i) generating TSFs
  - (ii) generating intermediate questions
  
We already provide the annotated TSF, and intermediate questions for TIQ and TimeQuestions benchmarks, as training data respectively.
    If you would like to generate annotated TSF for other datasets, please follow the instruction in:
```    
faith/distant_supervision/README.md
```
For generating intermediate questions as training data via GPT, please follow the instruction in:
```
faith/temporal_qu/implicit_resolver/seq2seq_iques/data_creation/README.md" 
```

#### FER
In the FER stage, we apply a [BERT-based](https://sbert.net) reranker to train classifier and score evidences as the input for answering.

#### EHA
In the EHA stage, there are two GNN models for:
- (i) graph reduction
- (ii) answer inference

We apply the two GNN models in answer inference stage:
 - (i) graph reduction: as the number of evidence decreases, the size of the graph is reduced. The number of evidence decreases from 100 to 20.
 - (ii) answer inference: among the 20 evidence, we conduct the answer inference and output the 5 evidence.

#### For training the models individually, please run the following script:

```bash
  bash scripts/pipeline.sh --train_<stage name> <PATH_TO_CONFIG>
```

For example, training the answer inference model.

```bash
  bash scripts/pipeline.sh --train_ha config/train_ha_answer_inference.yml
```

Note that you need two the config files for training the two GNN models respectively.


[//]: # (## Feedback)

[//]: # (The FAITH project by [Zhen Jia]&#40;zjia@swjtu.edu.cn&#41;, [Philipp Christmann]&#40;pchristm@mpi-inf.mpg.de&#41; and [Gerhard Weikum]&#40;weikum@mpi-inf.mpg.de&#41; is licensed under [MIT license]&#40;&#41;.)

[//]: # (## License)

[//]: # (The FAITH project by [Zhen Jia]&#40;zjia@swjtu.edu.cn&#41;, [Philipp Christmann]&#40;pchristm@mpi-inf.mpg.de&#41; and [Gerhard Weikum]&#40;weikum@mpi-inf.mpg.de&#41; is licensed under [MIT license]&#40;&#41;.)


