{
    "dataset_reader": {
        "type": "textcat",
        "debug": false,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            },
	        "elmo": {
		        "type": "elmo_characters",
		    }
        },
    },
  "datasets_for_vocab_creation": ["train"],
  "train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/ag-news/train.jsonl",
  "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/ag-news/dev.jsonl",
    "model": {
        "type": "seq2seq_classifier",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": true
                },
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                    "do_layer_norm": false,
                    "dropout": 0.2
                }
            }
        },
        "encoder": {
           "type": "lstm",
           "num_layers": 1,
           "bidirectional": true,
	       "input_size": 1324,
           "hidden_size": 128, 
        },
        "aggregations": ["maxpool", "final_state"],
        "output_feedforward": {
            "input_dim": 256,
            "num_layers": 1,
            "hidden_dims": 512,
            "activations": "relu",
            "dropout": 0.5
        },
        "classification_layer": {
            "input_dim": 512,
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
            "lr": 0.0004
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 75,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        }
    }
}
