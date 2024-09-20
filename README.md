# RE-traceability-tool

This repository contains scripts for recovering traceability links between issues and commits in GitHub projects using the T-BERT model. The repository provides tools for both replicating the results from our paper and applying the method to new projects of your own.

## Getting started

Clone the repository and install the required dependencies
```
git clone https://github.com/risha-parveen/RE-traceability-tool.git

cd RE-traceability-tool

pip install -r requirement.txt
```

## Replicating results for the four analyzed projects

To replicate the results from our study on the four GitHub projects, follow these steps:

1. Download the preprocessed data and trained model used in the study from our Figshare repository:
    * Preprocessed GitHub data from the four projects: Download here
    * Fully trained T-BERT model: Download here
2. Save the downloaded files into appropriate directories within the repository:
    * Place the preprocessed data in a directory named data/
    * Store the trained model in models/ directory
3. Run the evaluation to recover traceability links from the test data of the four projects:
```
cd evaluation/single_model

python single_eval.py \
  --data_dir ../data \
  --model_dir ../models \
  --exp_name replication_study
```

## Testing on new projects
If you'd like to test the traceability tool on your own GitHub projects, follow these steps:

1. Collect and preprocess the GitHub data from your own repositories:

```
cd github

python data_process.py \
  --repo_path <your_github_repository_path> \
  --root_data_dir <your_preprocessed_data_directory>
```
  Replace `<your_github_repository_path>` with the local path to your GitHub repository and `<your_preprocessed_data_directory>` with the path where the preprocessed data should be stored.

2. Run the evaluation to recover traceability links from your data:

```
cd evaluation/single_model

python single_eval.py \
  --repo_path <your_github_repository_path> \
  --exp_name <your_experiment_name>
```
  Replace `<your_github_repository_path>` with the path to your repository and `<your_experiment_name>` with a custom name for your experiment.

## Training the model on different architectures
You can also train T-BERT using other architectures (e.g., Siamese, Twin) by following the training steps provided by the original T-BERT repository: TraceBERT GitHub.
