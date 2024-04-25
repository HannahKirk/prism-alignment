# The PRISM Alignment Project

<div style="text-align:center">
   <img src="https://github.com/HannahKirk/prism-alignment/blob/main/prism_splash.png" alt="Summary of PRISM" width="75%">
</div>

## Description
This repo contains the PRISM dataset and accompanying code for our paper: [The PRISM Alignment Project: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models](https://arxiv.org/abs/2404.16019).

PRISM is a dataset that maps the characteristics and stated preferences from a survey of diverse humans onto their ratings of real-time interactions with large language models (LLMs).

## Format of Released Data
We release two primary jsonl files for our dataset. All variables are documented and explained in the [Code Book](https://github.com/HannahKirk/prism-alignment/blob/main/prism_codebook.pdf).
1. **The Survey** (`data/survey.jsonl`): The survey where users answer questions such as their stated preferences for LLM behaviours, their familarity with LLMs, a self-description and some basic demographics. Each row is a single user in our dataset, identified by a `user_id`.
2. **The Conversations** (`data/conversations.jsonl`):  Each participants' multiple conversation trees with LLMs and associated feedback. Each row is a single conversation, identified by a `conversation_id`, that can be matched back to a participant's survey profile via the `user_id`. The conversation itself is stored as a list of dictionaries representing human and model turns in the `conversation_history` column, which broadly follows the format of widely used Chat APIs.

We appreciate that different analyses require different data formats. In order to save people time in wrangling the conversational data into different formats we also present a long-format. Please note these contain the same data, just presented differently.

3. **The Utterances** (`utterances.jsonl`): Each row is a single scored utterance (human input - model response - score). Each row has an `utterance_id` that can be mapped back to the conversation data using `conversation_id` or the survey using `user_id`. The model responses and scores per each user input are in _long format_. Because of this format, the user inputs will be repeated for the set of model responses in  a single interaction turn.

We also provide code (in `src/utils/data_loader.py`) for transforming the conversations to a _wide format_. That is, each row is now a single turn within a conversation. For the first interaction where up to four models respond, we have `model_{a/b/c/d}` as four distinct columns and `score_{a/b/c/d}` as another four columns. Note that for subsequent turns, the same model responds and there are only two responses so `model/score_{c/d}` will always be missing.

Finally, for every text instance in PRISM, we provide:

4. **The Metadata** (`data/metadata/metadata.jsonl`): Each row is a text instance with attached information on language detection, personal or private information (PII) detection and moderation flags.

## Licensing and Attribution
For a full **data clause**, see the [Code Book](https://github.com/HannahKirk/prism-alignment/blob/main/prism_codebook.pdf) or the [Supplementary materials in our paper](https://arxiv.org/abs/2404.16019).

Human-written texts (including prompts) within the dataset are licensed under the Creative Commons Attribution 4.0 International License (CC-BY-4.0). Model responses are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC-4.0). Use of model responses must abide by the original model provider licenses.

For proper attribution when using this dataset in any publications or research outputs, please cite our paper:
```
@misc{kirk2024PRISM,
      title={The PRISM Alignment Project: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models}, 
      author={Hannah Rose Kirk and Alexander Whitefield and Paul Röttger and Andrew Bean and Katerina Margatina and Juan Ciro and Rafael Mosquera and Max Bartolo and Adina Williams and He He and Bertie Vidgen and Scott A. Hale},
      year={2024},
      eprint={2404.16019},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Or cite our dataset (DOI = `10.57967/hf/2113`):

```
@misc {kirk2024PRISMdataset,
    author       = {Kirk, Hannah Rose and Whitefield, Alexander and Röttger, Paul and Bean, Andrew and Margatina, Katerina and Ciro, Juan and Mosquera, Rafael and Bartolo, Max and Williams, Adina and He, He and Vidgen, Bertie and Hale, Scott A.},
    title        = {The PRISM Alignment Dataset},
    year         = {2024},
    url          = {https://huggingface.co/datasets/HannahRoseKirk/prism-alignment},
    doi          = {10.57967/hf/2113},
    publisher    = {Hugging Face}
}
```

**Suggested In-text Citation:** Kirk, H. R., Whitefield, A., Röttger, P., Bean, A., Margatina, K., Ciro, J., Mosquera, R., Bartolo, M., Williams, A., He, H., Vidgen, B., & Hale, S. A. (2024). _The PRISM Alignment Dataset_. https://doi.org/10.57967/hf/2113.


## Code Details

The code in `./src` is coordinated and run in phases via `./scripts`. Any plotting or analysis is primarily conducted and presented in `./notebooks`. The clean data for release is in `./data`. We provide an export of our conda environment in `environment.yml`.

The code base was developed for our preprint and is still being prepared/refined for final submission. It is a work in progress! 👷

### Stages
We consider there to be two phases to the codebase for this project:

1. **Pre-release**: Any data collection, cleaning or processing steps we conduct prior to releasing the dataset publically. We release this code for transparency but not the interim data files in order to protect the privacy of our data subjects. These steps are sequentially run via `./scripts/pre_release_pipeline.sh`. Some steps of the processing (particularly census-matching and any manual annotation) are presented in `notebooks/pre-release processing`.
2. **Post-release**: The analysis presented in our preprint is based on the seperate investigations in `notebooks/analysis`.

## Issue Reporting
If there are any issues with the dataset, for example, the discovery of personal or private information (PII) or requests from participants for data removal, please report it to us via the [Issue Reporting Form](https://forms.gle/WFvqvDBNqbCteoZV7).
