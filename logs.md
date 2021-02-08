## Make datasets

```sh
# Answerability threshold = 0
$ mkdir -p ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0
$ python make_rcqa_dataset.py \
--dataset_file ~/data/tohoku-rcqa/all-v1.0.json.gz \
--output_dir ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0 \
--mode original \
--answerability_threshold 0
# Train dataset: 9691 questions
# Dev dataset: 1500 questions
# Test dataset: 1400 questions
# Processing training datasets
# Answerable: 43610, Not answerable: 0, Total: 43610
# Processing the development dataset
# Answerable: 6863, Not answerable: 0, Total: 6863
# Processing the test dataset
# Answerable: 6178, Not answerable: 0, Total: 6178
```

```sh
# Answerability threshold = 2
$ mkdir -p ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2
$ python make_rcqa_dataset.py \
--dataset_file ~/data/tohoku-rcqa/all-v1.0.json.gz \
--output_dir ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2 \
--mode original \
--answerability_threshold 2
# Train dataset: 9691 questions
# Dev dataset: 1500 questions
# Test dataset: 1400 questions
# Processing training datasets
# Answerable: 21091, Not answerable: 22519, Total: 43610
# Processing the development dataset
# Answerable: 3524, Not answerable: 3339, Total: 6863
# Processing the test dataset
# Answerable: 3079, Not answerable: 3099, Total: 6178
```

```sh
# Answerability threshold = 2, Answerable only
$ mkdir ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only
$ python make_rcqa_dataset.py \
--dataset_file ~/data/tohoku-rcqa/all-v1.0.json.gz \
--output_dir ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only \
--mode original \
--answerability_threshold 2 \
--answerable_only
# Train dataset: 9691 questions
# Dev dataset: 1500 questions
# Test dataset: 1400 questions
# Processing training datasets
# Answerable: 21091, Not answerable: 0, Total: 21091
# Processing the development dataset
# Answerable: 3524, Not answerable: 0, Total: 3524
# Processing the test dataset
# Answerable: 3079, Not answerable: 0, Total: 3079
```

```sh
# Answerability threshold = 2, Hard unanswerable
$ mkdir -p ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard
$ python make_rcqa_dataset.py \
--dataset_file ~/data/tohoku-rcqa/all-v1.0.json.gz \
--output_dir ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard \
--mode hard_unanswerable \
--es_index_name jawiki_20200831_paragraphs \
--retriever_k 5000 \
--answerability_threshold 2
# Train dataset: 9691 questions
# Dev dataset: 1500 questions
# Test dataset: 1400 questions
# Processing training datasets
# Answerable: 21091, Not answerable: 22519, Total: 43610
# Processing the development dataset
# Answerable: 3524, Not answerable: 3339, Total: 6863
# Processing the test dataset
# Answerable: 3079, Not answerable: 3099, Total: 6178
```

### For open-domain QA

```sh
$ ~/local/elasticsearch-6.5.4/bin/elasticsearch &

$ mkdir -p ~/work/open-book-qa/datasets/tohoku-rcqa/open-100
$ python make_rcqa_dataset.py \
--dataset_file ~/data/tohoku-rcqa/all-v1.0.json.gz \
--output_dir ~/work/open-book-qa/datasets/tohoku-rcqa/open-100-new \
--mode open_domain \
--es_index_name jawiki_20200831_paragraphs \
--retriever_k 100 \
--skip_train
# Train dataset: 9691 questions
# Dev dataset: 1500 questions
# Test dataset: 1400 questions
# Skipping the training dataset
# Processing the development dataset
# Answerable: 13808, Not answerable: 136192, Total: 150000
# Processing the test dataset
# Answerable: 12500, Not answerable: 127500, Total: 140000

$ mkdir -p ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k
$ python make_rcqa_dataset.py \
--dataset_file ~/data/tohoku-rcqa/all-v1.0.json.gz \
--output_dir ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k \
--mode open_domain \
--es_index_name jawiki_20200831_paragraphs \
--retriever_k 1000 \
--skip_train
# Train dataset: 9691 questions
# Dev dataset: 1500 questions
# Test dataset: 1400 questions
# Skipping the training dataset
# Processing the development dataset
# Answerable: 46398, Not answerable: 1453602, Total: 1500000
# Processing the test dataset
# Answerable: 44865, Not answerable: 1355135, Total: 1400000

$ mkdir ~/work/open-book-qa/datasets/tohoku-rcqa/open-10k
$ python make_rcqa_dataset.py \
--dataset_file ~/data/tohoku-rcqa/all-v1.0.json.gz \
--output_dir ~/work/open-book-qa/datasets/tohoku-rcqa/open-10k \
--mode open_domain \
--es_index_name jawiki_20200831_paragraphs \
--retriever_k 10000 \
--skip_train
# Train dataset: 9691 questions
# Dev dataset: 1500 questions
# Test dataset: 1400 questions
# Skipping the training dataset
# Processing the development dataset
# Answerable: 166098, Not answerable: 14833902, Total: 15000000
# Processing the test dataset
# Answerable: 166591, Not answerable: 13833409, Total: 14000000

$ mkdir ~/work/open-book-qa/datasets/tohoku-rcqa/open-100-with-title
$ python make_rcqa_dataset.py \
--dataset_file ~/data/tohoku-rcqa/all-v1.0.json.gz \
--output_dir ~/work/open-book-qa/datasets/tohoku-rcqa/open-100-with-title \
--mode open_domain \
--es_index_name jawiki_20200831_paragraphs \
--retriever_k 100 \
--skip_train
# Train dataset: 9691 questions
# Dev dataset: 1500 questions
# Test dataset: 1400 questions
# Skipping the training dataset
# Processing the development dataset
# Answerable: 15440, Not answerable: 134560, Total: 150000
# Processing the test dataset
# Answerable: 13984, Not answerable: 126016, Total: 140000

$ mkdir ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k-with-title
$ python make_rcqa_dataset.py \
--dataset_file ~/data/tohoku-rcqa/all-v1.0.json.gz \
--output_dir ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k-with-title \
--mode open_domain \
--es_index_name jawiki_20200831_paragraphs \
--retriever_k 1000 \
--skip_train
# Train dataset: 9691 questions
# Dev dataset: 1500 questions
# Test dataset: 1400 questions
# Skipping the training dataset
# Processing the development dataset
# Answerable: 50490, Not answerable: 1449510, Total: 1500000
# Processing the test dataset
# Answerable: 49141, Not answerable: 1350859, Total: 1400000

$ mkdir ~/work/open-book-qa/datasets/tohoku-rcqa/open-10k-with-title
$ python make_rcqa_dataset.py \
--dataset_file ~/data/tohoku-rcqa/all-v1.0.json.gz \
--output_dir ~/work/open-book-qa/datasets/tohoku-rcqa/open-10k-with-title \
--mode open_domain \
--es_index_name jawiki_20200831_paragraphs \
--retriever_k 10000 \
--skip_train
# Train dataset: 9691 questions
# Dev dataset: 1500 questions
# Test dataset: 1400 questions
# Skipping the training dataset
# Processing the development dataset
# Answerable: 176898, Not answerable: 14823102, Total: 15000000
# Processing the test dataset
# Answerable: 176978, Not answerable: 13823022, Total: 14000000
```

## Train GloVe embeddings for BiDAF models

```sh
$ cat ~/work/open-book-qa/glove/corpus_files/corpus.txt.*|mecab -O wakati -d ~/local/lib/mecab/dic/ipadic > ~/work/open-book-qa/glove/corpus_tokenized.txt
$ cd ~/work/open-book-qa/glove
$ wget https://nlp.stanford.edu/software/GloVe-1.2.zip
$ unzip GloVe-1.2.zip
$ cd GloVe-1.2
$ make
$ build/vocab_count -min-count 5 < ../corpus_tokenized.txt > ../vocab.txt
$ build/cooccur -memory 32.0 -vocab-file ../vocab.txt -window-size 15 < ../corpus_tokenized.txt > ../cooccurrence.bin
$ build/shuffle -memory 32.0 < ../cooccurrence.bin > ../cooccurrence.shuf.bin
$ build/glove -save-file ../vectors_100d -threads 8 -input-file ../cooccurrence.shuf.bin -x-max 10 -iter 15 -vector-size 100 -binary 2 -vocab-file ../vocab.txt
```

## Training

### BiDAF

```sh
# Answerability threshold = 0
$ allennlp train configs/tohoku-rcqa/threshold-0/bidaf.jsonnet \
--serialization-dir ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf \
--include-package allennlp_modules

# Answerability threshold = 2
$ allennlp train configs/tohoku-rcqa/threshold-2/bidaf.jsonnet \
--serialization-dir ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf \
--include-package allennlp_modules

# Answerability threshold = 2, answerable only
$ allennlp train configs/tohoku-rcqa/threshold-2_answerable-only/bidaf.jsonnet \
--serialization-dir ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf \
--include-package allennlp_modules

# Answerability threshold = 2, hard unanswerable
$ allennlp train configs/tohoku-rcqa/threshold-2_hard/bidaf.jsonnet \
--serialization-dir ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bidaf \
--include-package allennlp_modules
```

### BERT-base

```sh
# Answerability threshold = 0
$ allennlp train configs/tohoku-rcqa/threshold-0/bert-base.jsonnet \
--serialization-dir ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base \
--include-package allennlp_modules

# Answerability threshold = 2
$ allennlp train configs/tohoku-rcqa/threshold-2/bert-base.jsonnet \
--serialization-dir ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base \
--include-package allennlp_modules

# Answerability threshold = 2, answerable only
$ allennlp train configs/tohoku-rcqa/threshold-2_answerable-only/bert-base.jsonnet \
--serialization-dir ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base \
--include-package allennlp_modules

# Answerability threshold = 2, hard unanswerable
$ allennlp train configs/tohoku-rcqa/threshold-2_hard/bert-base.jsonnet \
--serialization-dir ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base \
--include-package allennlp_modules
```

### BERT-base v2

```sh
# Answerability threshold = 0
$ allennlp train configs/tohoku-rcqa/threshold-0/bert-base-v2.jsonnet \
--serialization-dir ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2 \
--include-package allennlp_modules

# Answerability threshold = 2
$ allennlp train configs/tohoku-rcqa/threshold-2/bert-base-v2.jsonnet \
--serialization-dir ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2 \
--include-package allennlp_modules

# Answerability threshold = 2, answerable only
$ allennlp train configs/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2.jsonnet \
--serialization-dir ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2 \
--include-package allennlp_modules

# Answerability threshold = 2, hard unanswerable
$ allennlp train configs/tohoku-rcqa/threshold-2_hard/bert-base-v2.jsonnet \
--serialization-dir ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2 \
--include-package allennlp_modules
```

## Evaluation

### BiDAF

#### Dev set evaluation

```sh
# Answerability threshold = 0
$ python predict_bidaf.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0/dev-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/prediction_dev.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/prediction_dev.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/evaluation_dev.json \
--output_accuracies_per_score
# Total: 6863
# Reading Exact Match: 48.7%
# Reading Character-level F1: 59.4%
# Accuracy for Score 0: 42.1%
# Accuracy for Score 1: 44.8%
# Accuracy for Score 2: 52.2%
# Accuracy for Score 3: 52.4%
# Accuracy for Score 4: 55.0%
# Accuracy for Score 5: 56.1%

# Answerability threshold = 2
$ python predict_bidaf.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2/dev-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/prediction_dev.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/prediction_dev.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/evaluation_dev.json \
--output_answerability_scores \
--output_accuracies_per_score
# Total: 3524
# Reading Exact Match: 44.4%
# Reading Character-level F1: 51.4%
# Answerable Binary Precision: 78.9%
# Answerable Binary Recall: 66.9%
# Answerable Binary F1: 72.4%
# Unanswerable Binary Precision: 69.9%
# Unanswerable Binary Recall: 81.1%
# Unanswerable Binary F1: 75.1%
# Accuracy for Score 0: 85.9%
# Accuracy for Score 1: 70.0%
# Accuracy for Score 2: 30.9%
# Accuracy for Score 3: 39.9%
# Accuracy for Score 4: 48.3%
# Accuracy for Score 5: 54.6%

# Answerability threshold = 2, answerable only
$ python predict_bidaf.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/dev-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/prediction_dev.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/prediction_dev.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/evaluation_dev.json \
--output_accuracies_per_score
# Total: 3524
# Reading Exact Match: 58.7%
# Reading Character-level F1: 69.9%
# Accuracy for Score 0: 0.0%
# Accuracy for Score 1: 0.0%
# Accuracy for Score 2: 53.2%
# Accuracy for Score 3: 56.4%
# Accuracy for Score 4: 61.6%
# Accuracy for Score 5: 61.9%

# Answerability threshold = 2, hard unanswerable
$ python predict_bidaf.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/dev-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bidaf/prediction_dev.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bidaf/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bidaf/prediction_dev.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bidaf/evaluation_dev.json \
--output_answerability_scores \
--output_accuracies_per_score
# Total: 3524
# Reading Exact Match: 42.3%
# Reading Character-level F1: 47.6%
# Answerable Binary Precision: 83.0%
# Answerable Binary Recall: 59.9%
# Answerable Binary F1: 69.6%
# Unanswerable Binary Precision: 67.3%
# Unanswerable Binary Recall: 87.1%
# Unanswerable Binary F1: 75.9%
# Accuracy for Score 0: 87.5%
# Accuracy for Score 1: 86.2%
# Accuracy for Score 2: 34.7%
# Accuracy for Score 3: 39.3%
# Accuracy for Score 4: 47.5%
# Accuracy for Score 5: 46.0%
```

##### Plot EM per score

```sh
$ python plot_em_per_score.py \
--all_answerable_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/evaluation_dev.json \
--answerable_only_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/evaluation_dev.json \
--soft_answerability_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/evaluation_dev.json \
--output_file figs/bidaf_dev_em_per_score.pdf
```

#### Test set evaluation

```sh
# Answerability threshold = 0
$ python predict_bidaf.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0/test-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/prediction_test.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/prediction_test.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/evaluation_test.json \
--output_accuracies_per_score
# Total: 6178
# Reading Exact Match: 48.9%
# Reading Character-level F1: 58.5%
# Accuracy for Score 0: 42.2%
# Accuracy for Score 1: 46.7%
# Accuracy for Score 2: 54.0%
# Accuracy for Score 3: 56.5%
# Accuracy for Score 4: 53.6%
# Accuracy for Score 5: 53.3%

# Answerability threshold = 2
$ python predict_bidaf.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2/test-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/prediction_test.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/prediction_test.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/evaluation_test.json \
--output_answerability_scores \
--output_accuracies_per_score
# Total: 3079
# Reading Exact Match: 43.7%
# Reading Character-level F1: 49.0%
# Answerable Binary Precision: 76.9%
# Answerable Binary Recall: 65.1%
# Answerable Binary F1: 70.5%
# Unanswerable Binary Precision: 69.9%
# Unanswerable Binary Recall: 80.6%
# Unanswerable Binary F1: 74.9%
# Accuracy for Score 0: 85.3%
# Accuracy for Score 1: 69.5%
# Accuracy for Score 2: 31.8%
# Accuracy for Score 3: 40.4%
# Accuracy for Score 4: 46.2%
# Accuracy for Score 5: 53.1%

# Answerability threshold = 2, answerable only
$ python predict_bidaf.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/test-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/prediction_test.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/prediction_test.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/evaluation_test.json \
--output_accuracies_per_score
# Total: 3079
# Reading Exact Match: 60.6%
# Reading Character-level F1: 70.0%
# Accuracy for Score 0: 0.0%
# Accuracy for Score 1: 0.0%
# Accuracy for Score 2: 55.9%
# Accuracy for Score 3: 59.0%
# Accuracy for Score 4: 62.1%
# Accuracy for Score 5: 64.0%

# Answerability threshold = 2, hard unanswerable
$ python predict_bidaf.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/test-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bidaf/prediction_test.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bidaf/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bidaf/prediction_test.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bidaf/evaluation_test.json \
--output_answerability_scores \
--output_accuracies_per_score
# Total: 3079
# Reading Exact Match: 41.6%
# Reading Character-level F1: 46.0%
# Answerable Binary Precision: 80.8%
# Answerable Binary Recall: 56.8%
# Answerable Binary F1: 66.7%
# Unanswerable Binary Precision: 66.9%
# Unanswerable Binary Recall: 86.5%
# Unanswerable Binary F1: 75.4%
# Accuracy for Score 0: 86.5%
# Accuracy for Score 1: 86.6%
# Accuracy for Score 2: 38.0%
# Accuracy for Score 3: 37.0%
# Accuracy for Score 4: 45.8%
# Accuracy for Score 5: 44.6%
```

### BERT-base

#### Dev set evaluation

```sh
# Answerability threshold = 0
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0/dev-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/prediction_dev.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/prediction_dev.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/evaluation_dev.json \
--output_accuracies_per_score
# Total: 6863
# Reading Exact Match: 71.9%
# Reading Character-level F1: 81.1%
# Accuracy for Score 0: 62.1%
# Accuracy for Score 1: 70.1%
# Accuracy for Score 2: 76.8%
# Accuracy for Score 3: 78.9%
# Accuracy for Score 4: 81.3%
# Accuracy for Score 5: 78.3%

# Answerability threshold = 2
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2/dev-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/prediction_dev.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/prediction_dev.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/evaluation_dev.json \
--output_answerability_scores \
--output_accuracies_per_score
# Total: 3524
# Reading Exact Match: 69.9%
# Reading Character-level F1: 77.7%
# Answerable Binary Precision: 80.2%
# Answerable Binary Recall: 86.8%
# Answerable Binary F1: 83.4%
# Unanswerable Binary Precision: 84.8%
# Unanswerable Binary Recall: 77.4%
# Unanswerable Binary F1: 80.9%
# Accuracy for Score 0: 85.7%
# Accuracy for Score 1: 58.3%
# Accuracy for Score 2: 55.4%
# Accuracy for Score 3: 66.0%
# Accuracy for Score 4: 77.2%
# Accuracy for Score 5: 77.4%

# Answerability threshold = 2, Answerable only
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/dev-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/prediction_dev.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/prediction_dev.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/evaluation_dev.json \
--output_accuracies_per_score
# Total: 3524
# Reading Exact Match: 79.9%
# Reading Character-level F1: 88.4%
# Accuracy for Score 0: 0.0%
# Accuracy for Score 1: 0.0%
# Accuracy for Score 2: 76.9%
# Accuracy for Score 3: 81.2%
# Accuracy for Score 4: 81.6%
# Accuracy for Score 5: 79.7%

# Answerability threshold = 2, hard unanswerable
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/dev-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base/prediction_dev.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base/prediction_dev.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base/evaluation_dev.json \
--output_answerability_scores \
--output_accuracies_per_score
# Total: 3524
# Reading Exact Match: 67.9%
# Reading Character-level F1: 75.1%
# Answerable Binary Precision: 89.4%
# Answerable Binary Recall: 82.6%
# Answerable Binary F1: 85.9%
# Unanswerable Binary Precision: 83.0%
# Unanswerable Binary Recall: 89.7%
# Unanswerable Binary F1: 86.2%
# Accuracy for Score 0: 90.6%
# Accuracy for Score 1: 87.5%
# Accuracy for Score 2: 59.1%
# Accuracy for Score 3: 66.6%
# Accuracy for Score 4: 72.8%
# Accuracy for Score 5: 71.3%
```

##### Plot EM per score

```sh
$ python plot_em_per_score.py \
--all_answerable_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/evaluation_dev.json \
--answerable_only_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/evaluation_dev.json \
--soft_answerability_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/evaluation_dev.json \
--output_file figs/bert-base_dev_em_per_score.pdf
```

#### Test set evaluation

```sh
# Answerability threshold = 0
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0/test-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/prediction_test.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/prediction_test.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/evaluation_test.json \
--output_accuracies_per_score
# Total: 6178
# Reading Exact Match: 73.2%
# Reading Character-level F1: 81.0%
# Accuracy for Score 0: 63.2%
# Accuracy for Score 1: 71.4%
# Accuracy for Score 2: 77.3%
# Accuracy for Score 3: 80.9%
# Accuracy for Score 4: 82.7%
# Accuracy for Score 5: 82.0%

# Answerability threshold = 2
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2/test-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/prediction_test.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/prediction_test.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/evaluation_test.json \
--output_answerability_scores \
--output_accuracies_per_score
# Total: 3079
# Reading Exact Match: 73.6%
# Reading Character-level F1: 78.5%
# Answerable Binary Precision: 80.1%
# Answerable Binary Recall: 86.7%
# Answerable Binary F1: 83.3%
# Unanswerable Binary Precision: 85.6%
# Unanswerable Binary Recall: 78.6%
# Unanswerable Binary F1: 82.0%
# Accuracy for Score 0: 86.7%
# Accuracy for Score 1: 59.6%
# Accuracy for Score 2: 57.8%
# Accuracy for Score 3: 71.5%
# Accuracy for Score 4: 79.0%
# Accuracy for Score 5: 82.2%

# Answerability threshold = 2, answerable only
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/test-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/prediction_test.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/prediction_test.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/evaluation_test.json \
--output_accuracies_per_score
# Total: 3079
# Reading Exact Match: 82.1%
# Reading Character-level F1: 88.9%
# Accuracy for Score 0: 0.0%
# Accuracy for Score 1: 0.0%
# Accuracy for Score 2: 77.6%
# Accuracy for Score 3: 82.1%
# Accuracy for Score 4: 83.9%
# Accuracy for Score 5: 84.0%

# Answerability threshold = 2, hard unanswerable
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/test-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base/prediction_test.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base/prediction_test.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base/evaluation_test.json \
--output_answerability_scores \
--output_accuracies_per_score
# Total: 3079
# Reading Exact Match: 69.3%
# Reading Character-level F1: 74.1%
# Answerable Binary Precision: 88.6%
# Answerable Binary Recall: 80.9%
# Answerable Binary F1: 84.6%
# Unanswerable Binary Precision: 82.5%
# Unanswerable Binary Recall: 89.7%
# Unanswerable Binary F1: 86.0%
# Accuracy for Score 0: 90.2%
# Accuracy for Score 1: 88.6%
# Accuracy for Score 2: 62.2%
# Accuracy for Score 3: 66.5%
# Accuracy for Score 4: 71.8%
# Accuracy for Score 5: 74.9%
```

### BERT-base v2

#### Dev set evaluation

```sh
# Answerability threshold = 0
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0/dev-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/prediction_dev.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/prediction_dev.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/evaluation_dev.json \
--output_accuracies_per_score
# Total: 6863
# Reading Exact Match: 74.5%
# Reading Character-level F1: 82.8%
# Accuracy for Score 0: 64.0%
# Accuracy for Score 1: 72.6%
# Accuracy for Score 2: 81.1%
# Accuracy for Score 3: 82.8%
# Accuracy for Score 4: 84.1%
# Accuracy for Score 5: 80.5%

# Answerability threshold = 2
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2/dev-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_dev.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_dev.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/evaluation_dev.json \
--output_answerability_scores \
--output_accuracies_per_score
# Total: 3524
# Reading Exact Match: 69.6%
# Reading Character-level F1: 75.8%
# Answerable Binary Precision: 83.1%
# Answerable Binary Recall: 82.6%
# Answerable Binary F1: 82.9%
# Unanswerable Binary Precision: 81.8%
# Unanswerable Binary Recall: 82.3%
# Unanswerable Binary F1: 82.0%
# Accuracy for Score 0: 89.0%
# Accuracy for Score 1: 66.9%
# Accuracy for Score 2: 51.9%
# Accuracy for Score 3: 66.2%
# Accuracy for Score 4: 77.5%
# Accuracy for Score 5: 78.7%

# Answerability threshold = 2, Answerable only
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/dev-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/prediction_dev.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/prediction_dev.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/evaluation_dev.json \
--output_accuracies_per_score
# Total: 3524
# Reading Exact Match: 82.7%
# Reading Character-level F1: 90.3%
# Accuracy for Score 0: 0.0%
# Accuracy for Score 1: 0.0%
# Accuracy for Score 2: 77.6%
# Accuracy for Score 3: 83.8%
# Accuracy for Score 4: 85.5%
# Accuracy for Score 5: 83.3%

# Answerability threshold = 2, hard unanswerable
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/dev-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/prediction_dev.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/prediction_dev.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/evaluation_dev.json \
--output_answerability_scores \
--output_accuracies_per_score
# Total: 3524
# Reading Exact Match: 69.0%
# Reading Character-level F1: 75.0%
# Answerable Binary Precision: 93.0%
# Answerable Binary Recall: 81.2%
# Answerable Binary F1: 86.7%
# Unanswerable Binary Precision: 82.5%
# Unanswerable Binary Recall: 93.5%
# Unanswerable Binary F1: 87.7%
# Accuracy for Score 0: 94.2%
# Accuracy for Score 1: 92.1%
# Accuracy for Score 2: 61.0%
# Accuracy for Score 3: 66.6%
# Accuracy for Score 4: 74.1%
# Accuracy for Score 5: 72.4%
```

##### Plot EM per score

```sh
$ python plot_em_per_score.py \
--all_answerable_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/evaluation_dev.json \
--answerable_only_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/evaluation_dev.json \
--soft_answerability_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/evaluation_dev.json \
--output_file figs/bert-base-v2_dev_em_per_score.pdf
```

#### Test set evaluation

```sh
# Answerability threshold = 0
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0/test-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/prediction_test.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-0/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/prediction_test.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/evaluation_test.json \
--output_accuracies_per_score
# Total: 6178
# Reading Exact Match: 76.1%
# Reading Character-level F1: 82.8%
# Accuracy for Score 0: 66.2%
# Accuracy for Score 1: 74.1%
# Accuracy for Score 2: 82.8%
# Accuracy for Score 3: 83.8%
# Accuracy for Score 4: 83.1%
# Accuracy for Score 5: 84.7%

# Answerability threshold = 2
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2/test-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_test.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_test.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/evaluation_test.json \
--output_answerability_scores \
--output_accuracies_per_score
# Total: 3079
# Reading Exact Match: 70.8%
# Reading Character-level F1: 75.5%
# Answerable Binary Precision: 83.1%
# Answerable Binary Recall: 81.5%
# Answerable Binary F1: 82.3%
# Unanswerable Binary Precision: 82.0%
# Unanswerable Binary Recall: 83.6%
# Unanswerable Binary F1: 82.8%
# Accuracy for Score 0: 90.2%
# Accuracy for Score 1: 68.0%
# Accuracy for Score 2: 52.0%
# Accuracy for Score 3: 68.0%
# Accuracy for Score 4: 77.3%
# Accuracy for Score 5: 81.4%

# Answerability threshold = 2, Answerable only
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/test-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/prediction_test.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/prediction_test.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/evaluation_test.json \
--output_accuracies_per_score
# Total: 3079
# Reading Exact Match: 84.2%
# Reading Character-level F1: 90.2%
# Accuracy for Score 0: 0.0%
# Accuracy for Score 1: 0.0%
# Accuracy for Score 2: 81.2%
# Accuracy for Score 3: 85.1%
# Accuracy for Score 4: 84.0%
# Accuracy for Score 5: 85.7%

# Answerability threshold = 2, hard unanswerable
$ python predict_bert.py \
--input-file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/test-v2.0.json \
--output-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/prediction_test.json \
--model-file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/model.tar.gz \
--cuda-device 0
$ python evaluate_rcqa.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/prediction_test.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/evaluation_test.json \
--output_answerability_scores \
--output_accuracies_per_score
# Total: 3079
# Reading Exact Match: 70.9%
# Reading Character-level F1: 75.0%
# Answerable Binary Precision: 91.6%
# Answerable Binary Recall: 80.0%
# Answerable Binary F1: 85.4%
# Unanswerable Binary Precision: 82.4%
# Unanswerable Binary Recall: 92.7%
# Unanswerable Binary F1: 87.2%
# Accuracy for Score 0: 92.9%
# Accuracy for Score 1: 92.4%
# Accuracy for Score 2: 63.8%
# Accuracy for Score 3: 68.9%
# Accuracy for Score 4: 73.0%
# Accuracy for Score 5: 75.8%
```


## Evaluation - Open-Domain QA

### BiDAF

```sh
# Answerability threshold = 0
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-100/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/prediction_dev_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/model.tar.gz scripts/raiden/predict_bidaf.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-100/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/prediction_dev_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/metrics_dev_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bidaf/qa_prediction_dev_open.json \
--top_k 100
# Best EM: 0.2493 (k = 11)

# Answerability threshold = 2
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-100/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/prediction_dev_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/model.tar.gz scripts/raiden/predict_bidaf.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-100/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/prediction_dev_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/metrics_dev_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bidaf/qa_prediction_dev_open.json \
--top_k 100
# Best EM: 0.3193 (k = 25)

# Answerability threshold = 2, Answerable only
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-100/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/prediction_dev_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/model.tar.gz scripts/raiden/predict_bidaf.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-100/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/prediction_dev_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/metrics_dev_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bidaf/qa_prediction_dev_open.json \
--top_k 100
# Best EM: 0.3053 (k = 13)
```

### BERT-base

**Note:** Evalution on 1500 * 1000 question-document pairs takes 3h for preprocessing and 3h for prediction.

```sh
# Answerability threshold = 0
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-100/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/prediction_dev_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-100/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/prediction_dev_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/metrics_dev_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base/qa_prediction_dev_open.json \
--top_k 100
# Best EM: 0.4073 (k = 15)

# Answerability threshold = 2
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-100/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/prediction_dev_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-100/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/prediction_dev_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/metrics_dev_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base/qa_prediction_dev_open.json \
--top_k 100
# Best EM: 0.5100 (k = 70)

# Answerability threshold = 2, Answerable only
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-100/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/prediction_dev_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-100/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/prediction_dev_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/metrics_dev_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base/qa_prediction_dev_open.json \
--top_k 100
# Best EM: 0.4133 (k = 14)
```

### BERT-base v2

**Note:** Evalution on 1500 * 1000 question-document pairs takes 3h for preprocessing and 3h for prediction.

#### Dev set evaluation

```sh
# Answerability threshold = 0
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-1k/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/prediction_dev_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/prediction_dev_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/metrics_dev_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/qa_prediction_dev_open.json \
--top_k 1000
# Best EM: 0.421 (F1 = 0.522, k = 19)

# Answerability threshold = 2
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-1k/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_dev_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_dev_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/metrics_dev_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/qa_prediction_dev_open.json \
--top_k 1000
# Best EM: 0.551 (F1 = 0.657, k = 960)

# Answerability threshold = 2, 10k documents
$ qsub -jc gpu-container_g1.168h -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-10k/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_dev_open_10k.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-10k/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_dev_open_10k.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/metrics_dev_open_10k.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/qa_prediction_dev_open_10k.json \
--top_k 10000
# Best EM:

# Answerability threshold = 2, answerable only
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-1k/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/prediction_dev_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/prediction_dev_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/metrics_dev_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/qa_prediction_dev_open.json \
--top_k 1000
# Best EM: 0.427 (F1 = 0.529, k = 12)

# Answerability threshold = 2, hard unanswerable
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-1k/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/prediction_dev_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/prediction_dev_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/metrics_dev_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/qa_prediction_dev_open.json \
--top_k 1000
# Best EM: 0.511 (F1 = 0.624, k = 203)
```

##### cf. Retrieve documents WITH titles

```sh
# Answerability threshold = 0
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-1k-with-title/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/prediction_dev_open_with_title.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k-with-title/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/prediction_dev_open_with_title.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/metrics_dev_open_with_title.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/qa_prediction_dev_open_with_title.json \
--top_k 1000
# Best EM: 0.438 (F1 = 0.548, k = 22)

# Answerability threshold = 2
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-1k-with-title/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_dev_open_with_title.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k-with-title/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_dev_open_with_title.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/metrics_dev_open_with_title.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/qa_prediction_dev_open_with_title.json \
--top_k 1000
# Best EM: 0.558 (F1 = 0.668, k = 377)

# Answerability threshold = 2, Answerable only
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-1k-with-title/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/prediction_dev_open_with_title.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k-with-title/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/prediction_dev_open_with_title.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/metrics_dev_open_with_title.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/qa_prediction_dev_open_with_title.json \
--top_k 1000
# Best EM: 0.444 (F1 = 0.558, k = 9)

# Answerability threshold = 2, hard unanswerable
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-1k-with-title/dev-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/prediction_dev_open_with_title.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k/dev-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/prediction_dev_open_with_title.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/metrics_dev_open_with_title.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/qa_prediction_dev_open_with_title.json \
--top_k 1000
# Best EM:
```

##### Plot k-EM curve

```sh
$ python plot_k_em_curve.py \
--all_answerable_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/metrics_dev_open.txt \
--answerable_only_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/metrics_dev_open.txt \
--soft_answerability_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/metrics_dev_open.txt \
--hard_answerability_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/metrics_dev_open.txt \
--output_file figs/k_em_curve.pdf

# $ python plot_k_em_curve.py \
# --all_answerable_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/metrics_dev_open_with_title.txt \
# --answerable_only_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/metrics_dev_open_with_title.txt \
# --judge_answerability_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/metrics_dev_open_with_title.txt \
# --output_file figs/k_em_curve_with_title.pdf
```

#### Test set evaluation

```sh
# Answerability threshold = 0
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-1k/test-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/prediction_test_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/prediction_test_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/metrics_test_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-0/bert-base-v2/qa_prediction_test_open.json \
--top_k 19 \
--fix_k
# Best EM: 0.386 (F1 = 0.504, k = 19)

# Answerability threshold = 2
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-1k/test-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_test_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_test_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/metrics_test_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/qa_prediction_test_open.json \
--top_k 960 \
--fix_k
# Best EM: 0.520 (F1 = 0.647, k = 960)

# Answerability threshold = 2, 10k documents
$ qsub -jc gpu-container_g1.168h -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-10k/test-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_test_open_10k.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-10k/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/prediction_test_open_10k.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/metrics_test_open_10k.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2/bert-base-v2/qa_prediction_test_open_10k.json \
--top_k 10000
# Best EM:

# Answerability threshold = 2, Answerable only
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-1k/test-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/prediction_test_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/prediction_test_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/metrics_test_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_answerable-only/bert-base-v2/qa_prediction_test_open.json \
--top_k 12 \
--fix_k
# Best EM: 0.391 (F1 = 0.508, k = 12)

# Answerability threshold = 2, Hard unanswerable
$ qsub -v INPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/datasets/tohoku-rcqa/open-1k/test-v2.0.json,OUTPUT_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/prediction_test_open.json,MODEL_FILE=/uge_mnt/home/m-suzuki/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/model.tar.gz scripts/raiden/predict_bert.sh
$ python evaluate_rcqa_open.py \
--dataset_file ~/work/open-book-qa/datasets/tohoku-rcqa/open-1k/test-v2.0.json \
--prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/prediction_test_open.json \
--output_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/metrics_test_open.txt \
--output_qa_prediction_file ~/work/open-book-qa/training/tohoku-rcqa/threshold-2_hard/bert-base-v2/qa_prediction_test_open.json \
--top_k 203 \
--fix_k
# Best EM: Best EM: 0.498 (F1 = 0.617, k = 203)
```
