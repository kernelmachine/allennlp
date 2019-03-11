import torch
from overrides import overrides
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.util import masked_max



@Seq2VecEncoder.register("maxpool")
class MaxPoolEncoder(Seq2VecEncoder):
    """
    A ``MaxPoolEncoder`` is a simple :class:`Seq2VecEncoder` which maxpools the embeddings
    of a sequence across the time dimension.
    The input to this module is of shape ``(batch_size, num_tokens, embedding_dim)``,
    and the output is of shape ``(batch_size, embedding_dim)``.

    Parameters
    ----------
    embedding_dim: ``int``
        This is the input dimension to the encoder.
    """
    def __init__(self,
                 embedding_dim: int) -> None:
        super(MaxPoolEncoder, self).__init__()
        self._embedding_dim = embedding_dim

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor=None):  #pylint: disable=arguments-differ
        if mask is not None:
            # replace the values we don't care about with a very small number, so it's effectively ignored
            # in the maxpool.
            broadcast_mask = mask.unsqueeze(-1).float()
            one_minus_mask = (1.0 - broadcast_mask).byte()
            tokens = tokens.masked_fill(one_minus_mask, -1e-7)
        maxpool, _ = tokens.max(dim=1, keepdim=False)
        return maxpool
