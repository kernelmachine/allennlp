import torch
from allennlp.modules import TokenEmbedder
from typing import Dict
from allennlp.data import Vocabulary
from allennlp.common import Params, Tqdm

@TokenEmbedder.register("onehot_token_embedder")
class OnehotTokenEmbedder(TokenEmbedder):
    """
    """
    def __init__(self, num_embeddings: int, projection_dim: int = None) -> None:
        super(OnehotTokenEmbedder, self).__init__()
        self.num_embeddings = num_embeddings
        if projection_dim:
            self._projection = torch.nn.Linear(num_embeddings, projection_dim)
        else:
            self._projection = None
        self.output_dim = projection_dim or num_embeddings

    def get_output_dim(self):
        return self.num_embeddings

    def compute_bow(self, tokens: torch.IntTensor, vocab_size: int) -> torch.Tensor:
        """
        Compute a bag of words representation (matrix of size NUM_DOCS X VOCAB_SIZE) of tokens.

        Params
        ______
        tokens : ``Dict[str, torch.Tensor]``
            tokens to compute BOW of
        index_to_token_vocabulary : ``Dict``
            vocabulary mapping index to token
        stopword_indicator: torch.Tensor, optional
            onehot tensor of size 1 x VOCAB_SIZE, indicating words in vocabulary to ignore when 
            generating BOW representation.
        """
        bow_vectors = []
        mask = get_text_field_mask({'tokens': tokens})
        for document, doc_mask in zip(tokens, mask):
            document = torch.masked_select(document, doc_mask.byte())
            vec = torch.bincount(document, minlength=vocab_size).float()
            vec = vec.view(1, -1)
            bow_vectors.append(vec)
        return torch.cat(bow_vectors, 0)

    def forward(self, # pylint: disable=arguments-differ
                inputs: torch.Tensor) -> torch.Tensor:
        bow_output = self.compute_bow(inputs, self.num_embeddings)
        if self._projection:
            projection = self._projection
            bow_output = projection(bow_output)
        return bow_output

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'Embedding':  # type: ignore
        """
        we look for a ``vocab_namespace`` key in the parameter dictionary to know which vocabulary to use.
        """
        # pylint: disable=arguments-differ
        vocab_namespace = params.pop("vocab_namespace", "tokens")
        num_embeddings = vocab.get_vocab_size(vocab_namespace)
        projection_dim = params.pop_int("projection_dim", None)
        params.assert_empty(cls.__name__)
        return cls(num_embeddings=num_embeddings,
                   projection_dim=projection_dim)