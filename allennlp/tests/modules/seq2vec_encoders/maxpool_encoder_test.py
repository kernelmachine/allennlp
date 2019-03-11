# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.modules.seq2vec_encoders import MaxPoolEncoder
from allennlp.common.testing import AllenNlpTestCase

class TestMaxPoolEncoder(AllenNlpTestCase):
    def test_get_dimension_is_correct(self):
        encoder = MaxPoolEncoder(embedding_dim=5)
        assert encoder.get_input_dim() == 5
        assert encoder.get_output_dim() == 5
        encoder = MaxPoolEncoder(embedding_dim=12)
        assert encoder.get_input_dim() == 12
        assert encoder.get_output_dim() == 12

    def test_can_construct_from_params(self):
        params = Params({
                'embedding_dim': 5,
                })
        encoder = MaxPoolEncoder.from_params(params)
        assert encoder.get_input_dim() == 5
        assert encoder.get_output_dim() == 5
        params = Params({
                'embedding_dim': 12,
                })
        encoder = MaxPoolEncoder.from_params(params)
        assert encoder.get_input_dim() == 12
        assert encoder.get_output_dim() == 12

    def test_forward_does_correct_computation(self):
        encoder = MaxPoolEncoder(embedding_dim=2)
        input_tensor = torch.FloatTensor([[[.7, .8], [.1, 1.5], [.3, .6]], [[.5, 1.3], [1.4, 1.1], [.3, 1.6]]])
        mask = torch.ByteTensor([[1, 1, 1], [1, 1, 0]])
        encoder_output = encoder(input_tensor, mask)
        assert_almost_equal(encoder_output.data.numpy(),
                            numpy.asarray([[.7, 1.5], [1.4, 1.3]]))


    def test_forward_does_correct_computation_no_mask(self):
        encoder = MaxPoolEncoder(embedding_dim=2)
        input_tensor = torch.FloatTensor([
                [[.7, .8], [.1, 1.5], [.3, .6]], [[.5, .3], [1.4, 1.1], [.3, .9]]
        ])
        encoder_output = encoder(input_tensor)
        assert_almost_equal(encoder_output.data.numpy(),
                            numpy.asarray([[.7, 1.5],
                                           [1.4, 1.1]]))
