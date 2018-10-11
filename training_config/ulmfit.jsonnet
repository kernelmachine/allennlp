		{
    "dataset_reader": {
        "type": "ag",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        },
    },
  "train_data_path": "s3://suching-dev/ag_news_csv/train.csv",
  "validation_data_path": "s3://suching-dev/ag_news_csv/test.csv",
    "model": {
        "type": "infersent",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
                    "embedding_dim": 300,
                    "trainable": true
                }
            }
        },
        "encoder": {
           "type": "lstm",
           "num_layers": 1,
           "bidirectional": true,
	   "input_size": 300,
           "hidden_size": 300, 
        },
        "output_feedforward": {
            "input_dim": 600,
            "num_layers": 1,
            "hidden_dims": 300,
            "activations": "relu",
            "dropout": 0.5
        },
        "output_logit": {
            "input_dim": 300,
            "num_layers": 1,
            "hidden_dims": 4,
            "activations": "linear"
        },
        "initializer": [
            [".*linear_layers.*weight", {"type": "xavier_uniform"}],
            [".*linear_layers.*bias", {"type": "zero"}],
            [".*weight_ih.*", {"type": "xavier_uniform"}],
            [".*weight_hh.*", {"type": "orthogonal"}],
            [".*bias_ih.*", {"type": "zero"}],
            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
	  "type": "adam",
          "lr": 0.0004,
          "parameter_groups": [
		[".*linear_layers.*weight", {}],
		[ ".*weight_hh.*", {}],
                [ ".*weight_ih.*", {}]
		]
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 75,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 75,
            "num_steps_per_epoch": 3750,
            "gradual_unfreezing": true,
            "discriminative_fine_tuning": true
        }
    }
}

