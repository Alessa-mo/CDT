# CDT
This repository provides an open-source implementation of the paper **CDT: A Comprehensive Capability Framework for Large Language Models Across Cognition, Domain, and Task.** It includes methods for annotating datasets with capability tags across the three dimensions, computing capability metrics for dataset evaluation, and selecting data in both general and specific scenarios.
<div align="center">
    <img src="assets/CDT.png" alt="image" width="500" />
</div>

## 🔗Links
Paper: [CDT: A Comprehensive Capability Framework for Large Language Models Across Cognition, Domain, and Task](todo)

You can download the tagging models from this link: [CDT](https://huggingface.co/Alessamo/models)

## 🗂️Directory Structure

- **[tag_annotate](tag_annotate)**: This folder contains the core components for tagging instructions based on the CDT framework.
  - **Prompt Files**: Prompts used by the tag annotator.
  - **Tagging Code**: Python scripts responsible for annotating instructions with capability tags.
  - **Tag Post-Processing**: Code for cleaning and refining the tags after they’ve been assigned.

- **[dataset_evaluation](dataset_evaluation)**: This folder includes methods for computing Balance and Coverage metrics.
  
- **[data_selection](data_selection)**: This folder includes methods for data selection in general and specific scenarios.
  - **General Scenario Selection**: Code for general scenario data selection.
  - **Special Scenario Selection**: Code for specific scenario data selection.

- **[data](data)**: This folder contains the data used in the paper, including the tagged data pool and the data selected by CDT in both scenarios.

- **[train](train)**: This folder contains the configuration files for training in our experiments.

## ⚙️Environment Setup

To install all the relevant packages, run the following:

```bash
conda create -n [environment_name] --file requirements.txt
conda activate [environment_name]
```

## 🚀Usage
### Data preparation
Make sure your data is a JSON file and has the following format:
```json
[
    {
        "messages": [
            {
                "role": "user",
                "content": "xxxx"
            },
            {
                "role": "assistant",
                "content": "xxxx"
            }
        ]
    },
]
```

### Tag annotation
To tag capability labels, run the following scripts:
```bash
cd tag_annotate
export CUDA_VISIBLE_DEVICES=0
python annotate.py \
    --data_path path/to/your/data \
    --output_dir path/to/output/dir \
    --model_path CDT_model_path \
    --prompt_file ./prompt/annotation_prompt.jsonl \
    --cognition_skill_file ./prompt/cognition.json \
    --domain_skill_file ./prompt/domain.json \
    --task_skill_file ./prompt/task.json \
    --tag_type "task" \
    --batch_size 32
```
Important parameters:
- `--model_path`: The path to the CDT tagging model.
- `--tag_type`: The type of tags to be assigned. It can be `"cognition"`, `"domain"`, or `"task"`.
- `--prompt_file`: The path to the prompt file.
- `--cognition_skill_file`: The path to the file that contains the cognition capabilities.
- `--domain_skill_file`: The path to the file that contains the domain capabilities.
- `--task_skill_file`: The path to the file that contains the task capabilities.

❗️**Notes**:

1.⚠️Ensure that `tag_type` and `model_path` match; for example, if `tag_type` is "task", select the corresponding task annotator.

2.The annotated data will be saved in {output_dir}/{data_name}_{tag_type}.json

Then run the following scripts to post-process the tags:
```bash
python postprocess_tag.py \
    --data_path path/to/the/annotated/data \
    --output_file path/to/output/file \
    --tag_type "task" \
    --cognition_skill_file ./prompt/cognition.json \
    --domain_skill_file ./prompt/domain.json \
    --task_skill_file ./prompt/task.json
```
❗️**Notes**:
⚠️Ensure that `data_path` and `tag_type` match; for example, if the data is annotated with task labels, then `tag_type` should be set to "task".

Finally, run the following script to merge all the tags:
```bash
python merge_tags.py \
    --cognition_path your/path/to/the/cleaned/data/annotated/with/cognition \
    --domain_path your/path/to/the/cleaned/data/annotated/with/domain \
    --task_path your/path/to/the/cleaned/data/annotated/with/task \
    --output_file path/to/output/file \
    --cognition_skill_file ./prompt/cognition.json \
    --domain_skill_file ./prompt/domain.json \
    --task_skill_file ./prompt/task.json
```
The output is a JSON file, formatted as follows:
```json
[
    {
    "messages": [
      {
        "role": "user",
        "content": "How can we reduce air pollution?"
      },
      {
        "role": "assistant",
        "content": "There are several ways to reduce air pollution, including:\n\n1. Reduce energy consumption: By conserving energy, we reduce the amount of pollution emitted from power plants...."
      }
    ],
    "tag": {
      "cognition": [
        "HP",
        "SP"
      ],
      "task": [
        "Open QA"
      ],
      "domain": [
        "Earth Science"
      ]
    }
  },
]
```
### Dataset Evaluation
First, use the preceding scripts to label the capabilities of the dataset you want to evaluate. Then, run the following scripts:
```bash
cd dataset_evaluation
python run_metrics.py \
        --data_path your/path/to/the/annotated/data \

```
Then it will print out the Coverage and Balance metrics.

<a id="data-selection"></a>
### Data Selection
#### Diversity Scenario
First, use the preceding scripts to label the capabilities of the data pool. Then, run the following scripts to select data for general scenarios based on diversity
```bash:
cd data_selection
python diversity.py \
    --data_path path/to/the/annotated/data/pool \
    --output_file path/to/output/file \
    --top_p 0.2
```
Important parameters:
- `--top_p`: The top-p threshold. It determines the proportion of the total data to select.
#### Specific Scenario
First, use the preceding scripts to label the capabilities of the data pool and the validation dataset. Then, run the following scripts to select data for specific scenarios based on the capability tags in validation set.
```bash:
cd data_selection
python specific.py \
    --train_file path/to/the/annotated/data/pool \
    --valid_file path/to/the/annotated/validation/data \
    --output_file path/to/output/file \
    --top_p 0.2
```
### Training
We use LLama Factory to train the model. Please refer to the [LLama Factory](https://github.com/hiyouga/LLaMA-Factory) repository for installation and usage instructions.
Before training, you need to specify the location of the dataset obtained in the [previous step](#data-selection) in the `dataset_info.json` file,  located in the `data` folder of the LLama-Factory repository. An example is as follows:
```json
{
    "dataset_name": {
    "file_name": "file_path",
    "formatting": "sharegpt",
    "columns": {
        "messages": "messages"
    },
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant"
    }
  },
}
```
Then, run the following scripts to train the model:
```bash
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train LLaMA-Factory/CDT.yaml
```
## 📜 Citation
If you find this work useful, please cite:
```bash
```

## 💡 Acknowledgments
We extend our gratitude to the following resources that contributed to this work:
- **Flask**: This work references insights from [Flask](https://github.com/kaistAI/FLASK).
- **LLaMA Factory**: The training pipeline leverages [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), which provided efficient implementation and infrastructure for model fine-tuning.