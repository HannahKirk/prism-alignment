---
license: cc
language:
- en
tags:
- alignment
- human-feedback
- ratings
- preferences
- ai-safety
- llm
- survey
- fine-grained
pretty_name: The PRISM Alignment Dataset
size_categories:
- 10K<n<100K
configs:
- config_name: survey
  data_files: "survey.jsonl"
- config_name: conversations
  data_files: "conversations.jsonl"
- config_name: utterances
  data_files: "utterances.jsonl"
- config_name: metadata
  data_files: "metadata.jsonl"
---

# Dataset Card for PRISM
PRISM is a diverse human feedback dataset for preference and value alignment in Large Language Models (LLMs).
It maps the characteristics and stated preferences of humans from a detailed survey onto their real-time interactions with LLMs and contextual preference ratings


## Dataset Details

There are two sequential stages: first, participants complete a **Survey** where they answer questions about their demographics and stated preferences, then proceed to the **Conversations** with LLMs, where they input prompts, rate responses and give fine-grained feedback in a series of multi-turn interactions.

We survey 1500 participants born in 75 countries, residing in 38 countries. The majority of these participants progress to the conversations phase (1,396, 93%). 
At the beginning of the conversation, a participant chooses from three conversation types: _Unguided_, _Values guided_ or _Controversy guided_. They then construct an opening prompt of their choosing.

We include 21 different LLMs in the backend of our interface (with a mix of open-access and commerical API models). Four of these LLMs are selected at random for the opening turn of the conversations. The participant rates the model responses on a sliding cardinal scale between Terrible (1) and Perfect (100). The conversation continues with the highest-rated LLM in subsequent turns of human prompts and model responses (between 2-22 turns).

After the first turn, the same model responds with an A and B response to the human prompt (sampled at a non-deterministic temperature).

After the participant ends the conversation, they give fine-grained feedback on model performance, why they rated one model higher than the others avaliable, and natural language open-ended feedback on the conversation as a whole.

Each participant is asked to have six conversations in total, equally split acorss conversation types (but there is some deviation from this quota).

In total, there are 8,011 conversation trees, and 68,371 rated utterances (human prompt - model response - score 3-tuple).

For more information on the dataset, please see our paper `The PRISM Alignment Project: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models` or the [Codebook](https://github.com/HannahKirk/prism-alignment/blob/main/prism_codebook.pdf) on our Github.

- **Curated by:** This project was primarily conducted and recieved ethics approval via the University of Oxford. The project was assisted by researchers at other various academic and industry institutions.
- **Funded by:** This project was awarded the MetaAI Dynabench Grant "Optimising feedback between humans-and-models-in-the-loop". For additional compute support, the project was awarded the Microsoft Azure Accelerating Foundation Model Research Grant. For additional annotation support, we received funding from the OpenPhil grant and NSF grant (IIS-2340345) via New York University. We also received in the form of research access or credits from OpenAI, Anthropic, Aleph Alpha, Google, HuggingFace and Cohere.
- **Language(s) (NLP):** The majority of the dataset is in English (99%) due to task and crowdworker specifications.
- **License:** Human-written texts (including prompts) within the dataset are licensed under the Creative Commons Attribution 4.0 International License (CC-BY-4.0). Model responses are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC-4.0). Use of model responses must abide by the original model provider licenses.

### Dataset Sources

- **Repository:** https://github.com/HannahKirk/prism-alignment
- **Paper:** [COMING SOON]
- **Website:** https://hannahkirk.github.io/prism-alignment/


## Dataset Structure

We release two primary jsonl files for our dataset. All variables are documented and explained in our [Code Book](https://github.com/HannahKirk/prism-alignment/blob/main/prism_codebook.pdf).
1. **The Survey** (`survey.jsonl`): The survey where users answer questions such as their stated preferences for LLM behaviours, their familarity with LLMs, a self-description and some basic demographics. Each row is a single user in our dataset, identified by a `user_id`.
2. **The Conversations** (`conversations.jsonl`):  Each participants' multiple conversation trees with LLMs and associated feedback. Each row is a single conversation, identified by a `conversation_id`, that can be matched back to a participant's survey profile via the `user_id`. The conversation itself is stored as a list of dictionaries representing human and model turns in the `conversation_history` column, which broadly follows the format of widely used Chat APIs.

We appreciate that different analyses require different data formats. In order to save people time in wrangling the conversational data into different formats we also present a long-format. Please note this contains the same data, just presented differently.

3. **The Utterances** (`utterances.jsonl`): Each row is a single scored utterance (human input - model response - score). Each row has an `utterance_id` that can be mapped back to the conversation data using `conversation_id` or the survey using `user_id`. The model responses and scores per each user input are in _long format_. Because of this format, the user inputs will be repeated for the set of model responses in  a single interaction turn.

Finally, for every text instance in PRISM, we provide:

4. **The Metadata** (`metadata.jsonl`): Each row is a text instance with attached information on language detection, personal or private information (PII) detection and moderation flags.


## Terms of Use

### Purpose
The Dataset is provided for the purpose of research and educational use in the field of natural language processing, conversational agents, social science and related areas; and can be used to develop or evaluate artificial intelligence, including Large Language Models (LLMs).

### Usage Restrictions
Users of the Dataset should adhere to the terms of use for a specific model when using its generated responses. This includes respecting any limitations or use case prohibitions set forth by the original model's creators or licensors.

### Content Warning
The Dataset contains raw conversations that may include content considered unsafe or offensive. Users must apply appropriate filtering and moderation measures when using this Dataset for training purposes to ensure the generated outputs align with ethical and safety standards.

### No Endorsement of Content
The conversations and data within this Dataset do not reflect the views or opinions of the Dataset creators, funders or any affiliated institutions. The dataset is provided as a neutral resource for research and should not be construed as endorsing any specific viewpoints.

### No Deanonymisation
The User agrees not to attempt to re-identify or de-anonymise any individuals or entities represented in the Dataset. This includes, but is not limited to, using any information within the Dataset or triangulating other data sources to infer personal identities or sensitive information.

### Limitation of Liability
The authors and funders of this Dataset will not be liable for any claims, damages, or other liabilities arising from the use of the dataset, including but not limited to the misuse, interpretation, or reliance on any data contained within.


## Data Statement

We provide a full data statement in our paper [ADD LINK]. There, we have detailed breakdowns of participant demographics and geographic information.

## Citation
**BibTeX:**
```
@article{Kirk2024PRISM,
  title   = {The PRISM Alignment Project: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models},
  author  = {Kirk, Hannah Rose and Whitefield, Alexander and Röttger, Paul and Bean, Andrew and Margatina, Katerina and Ciro, Juan and Mosquera, Rafael and Bartolo, Max and Williams, Adina and He, He and Vidgen, Bertie and Hale, Scott A.},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year    = {2024}
}
```
**APA:**
Kirk, H. R., Whitefield, A., Röttger, P., Bean, A., Margatina, K., Ciro, J., Mosquera, R., Bartolo, M., Williams, A., He, H., Vidgen, B., & Hale, S. A. (2024). The PRISM Alignment Project: What Participatory, Representative and Individualised human feedback reveals about the Subjective and Multicultural alignment of Large Language Models. arXiv preprint arXiv:xxxx.xxxxx.


## Dataset Card Authors

Hannah Rose Kirk (hannah.kirk@oii.ox.ac.uk)

## Issue Reporting
If there are any issues with the dataset, for example, the discovery of personal or private information (PII) or requests from participants for data removal, please report it to us via the [Issue Reporting Form](https://forms.gle/WFvqvDBNqbCteoZV7).
