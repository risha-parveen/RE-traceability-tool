# RE-traceability-tool

This repository contains scripts for recovering traceability links between issues and commits in Github projects. 

To get started, clone this repository and install the required dependencies

```
git clone https://github.com/risha-parveen/RE-traceability-tool.git

pip install -r requirement.txt
```
Training:

The fully trained Single-architecture T-BERT model can be downloaded from [https://drive.google.com/file/d/1JUT9tVYfzSC_ZQb-GaWkz8rq95-YUMBM/view?usp=drive_link](url) and saved to the local.

You can also train T-BERT on other architectures like Siamese or Twin architecture by following the training steps presented in [https://github.com/jinfenglin/TraceBERT](url)


Recovering trace links from Github projects:

You can download the preprocessed test data of github projects from the following link: [https://drive.google.com/drive/folders/1hFFzJzxsbv7zqYFmRV9Y12h0dUjYFXB_?usp=sharing](url)

To collect data from different repositories, run the following command to collect and preprocess the data.

```
cd github

python data_process.py \
  --repo_path <git_repository_path> \
  --root_data_dir <test_data_root_directory>
```

To recover links from the git data run the following command:

```
cd evaluation/single_model

python single_eval.py \
  --repo_path <git_repository_path> \
  --exp_name <experiment_name>
```
