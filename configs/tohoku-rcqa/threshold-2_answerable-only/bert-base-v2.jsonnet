local transformer_model_name = "/home/m-suzuki/work/transformers-models/bert-base-japanese-v2";
local lr = 5e-5;
local epochs = 3;
local batch_size = 8;
local num_gradient_accumulation_steps = 4;
local length_limit = 512;
local use_amp = false;

{
  "dataset_reader": {
    "type": "transformer_squad",
    "transformer_model_name": transformer_model_name,
    "length_limit": length_limit,
    "stride": 128,
    "skip_impossible_questions": false,
    "max_query_length": 64
  },
  "train_data_path": "~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/train-v2.0.json",
  "validation_data_path": "~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_answerable-only/dev-v2.0.json",
  "vocabulary": {
    "type": "empty",
  },
  "model": {
    "type": "transformer_qa",
    "transformer_model_name": transformer_model_name,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": batch_size,
    }
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": 0.0,
      "parameter_groups": [
        [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}],
      ],
      "lr": lr,
      "eps": 1e-8,
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": epochs,
      "cut_frac": 0.0,
    },
    "grad_clipping": 1.0,
    "num_epochs": epochs,
    "validation_metric": "+per_instance_em",
    "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
    "use_amp": use_amp,
  },
}
