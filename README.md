# Pointer-Guided Pre-Training

This repo contains the code and fine-tuning data for the paper "Pointer-Guided Pre-Training: Infusing Large Language Models with Paragraph-Level Contextual Awareness"

## Installation

1. Clone repository.
2. `cd` into cloned directory.
3. Create and activate a new virtual or miniconda python environment with python 3.11. E.g. for miniconda:
   ```bash
      conda create -n pointer-pre-training python=3.11
      conda activate pointer-pre-training
   ```
4. Install package via `pip install .`


## Usage

### Re-create Wikipedia pre-training corpus
1. Download the latest wikipedia xml dump in the language of your choice from https://dumps.wikimedia.org/backup-index.html. 
2. Extract all wikipedia article names from the dump by navigating to `bash cd scripts/data_processing` and running 
   ```bash
   python get_all_wiki_article_names.py --dump <path-to-wiki-dump> --out <path-to-output-dir>
   ```
3. Create the html wikipedia corpus via the official API by running
   ```bash
   python get_wiki_html_via_api.py --dump <path-to-wiki-dump> --out <path-to-output-dir> --lang <language-eg-en-or-de>
   ```
4. Parse the created html dump to extract the text and create the final pre-training corpus by running
   ```bash
   python wiki_parser.py --html-dump <path-to-html-dump> --out <path-to-output-file.jsonl.gz>
   

### Pointer-guided pre-training
For pre-training navigate to `cd scripts`, use `run_pipeline.py` and provide the pre-training config `pretrain_cfg.yml`. 
All pre-training configurations and hyper-parameter setups are handled by the configuration file.
To train on GPUs use 
```bash
python run_pipeline.py --conda-ids 0 1 2 --config pretrain_cfg.yml
```
The configurations for pre-training can be found in the paper.

Now we cannot share the pre-training datasets, so you have to create your own pre-training data.
In `llm/data/data_iterators.py` you can see how our data iterators are implemented and new datasets are registered (dict at the end of the file).

It is important that your iterator yields a list of samples per article/document (list[dict]) 
where each sample is a dict with the required key `raw_segments` (list[str]).


### Pre-trained Models, Tokenizer and fine-tuning datasets
We provide all our pre-trained models (see paper), our custom tokenizer and the open-source fine-tuning datasets 
via the following [Google Drive Link](https://drive.google.com/file/d/1PzWmUCcH419e_7-BHRocJ3KKnDtDEkZy/view?usp=sharing).

Make sure to download the data and copy it to the `data` directory in the cloned repo.



### Downstream Task fine-tuning
To fine-tune a pre-trained model on a downstream task navigate to `cd scripts` and run
```bash
python run_pipeline.py --conda-ids 0 1 2 --config pubmed_20k_cfg.yml
```
which trains a model on the PubMed 20k dataset and utilizes 3 GPUs if available and necessary.


### Config Files
The pretrain_cfg.yml and the cs_abstract_cfg.yml files are commented and there structure is identical to the remaining
files. The main file `run_pipeline.py` uses [FluidML](https://github.com/fluidml/fluidml) to run the pipeline efficiently using multiprocessing.
Consult [FluidMLs](https://github.com/fluidml/fluidml) Readme for more infos on how to configure the pipeline and how to run grid-searches.



## Citation
If you use this code or the pre-trained models in your research, please cite the following paper:
```bibtex
@inproceedings{hillebrand2024,
  title={Pointer-Guided Pre-Training: Infusing Large Language Models with Paragraph-Level Contextual Awareness},
  author={Hillebrand, Lars and Pradhan, Prabhupad and Bauckhage, Christian and Sifa, Rafet},
  booktitle={Proc. ECMLPKDD},
  year={2024}
}
```
