{
  "dataset_reader": {
    "type": "squad2",
    "tokenizer": {
      "type": "mecab"
    },
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": false
      },
      "token_characters": {
        "type": "characters",
        "character_tokenizer": {
          "start_tokens": ["<s>"],
          "end_tokens": ["</s>"]
        },
        "min_padding_length": 5
      }
    },
    "no_answer_token": ""
  },
  "train_data_path": "~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/train-v2.0.json",
  "validation_data_path": "~/work/open-book-qa/datasets/tohoku-rcqa/threshold-2_hard/dev-v2.0.json",
  "model": {
    "type": "bidaf",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "~/work/open-book-qa/glove/vectors_100d.txt",
          "embedding_dim": 100,
          "trainable": false
        },
        "token_characters": {
          "type": "character_encoding",
          "embedding": {
            "embedding_dim": 16
          },
          "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": 100,
            "ngram_filter_sizes": [5]
          },
          "dropout": 0.2
        }
      }
    },
    "num_highway_layers": 2,
    "phrase_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 200,
      "hidden_size": 100,
      "num_layers": 1
    },
    "matrix_attention": {
      "type": "linear",
      "combination": "x,y,x*y",
      "tensor_1_dim": 200,
      "tensor_2_dim": 200
    },
    "modeling_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 800,
      "hidden_size": 100,
      "num_layers": 2,
      "dropout": 0.2
    },
    "span_end_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 1400,
      "hidden_size": 100,
      "num_layers": 1
    },
    "dropout": 0.2
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 40
    }
  },
  "trainer": {
    "num_epochs": 20,
    "grad_norm": 5.0,
    "patience": 10,
    "validation_metric": "+f1",
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "optimizer": {
      "type": "adam",
      "betas": [0.9, 0.9]
    }
  }
}
