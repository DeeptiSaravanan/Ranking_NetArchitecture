"""An implementation of aNMM Model."""

import matchzoo as mz
import numpy as np
import tensorflow
import keras
from keras.activations import softmax
from keras.initializers import RandomUniform

from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces

#ANMM
class ANMM(BaseModel):
    """
    ANMM Model.

    Examples:
        >>> model = ANMM()
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(
            name='dropout_rate', value=0.1,
            desc="The dropout rate.",
            hyper_space=hyper_spaces.quniform(0, 1, 0.05)
        ))
        params.add(Param(
            name='num_layers', value=2,
            desc="Number of hidden layers in the MLP "
                 "layer."
        ))
        params.add(Param(
            name='hidden_sizes', value=[30, 30],
            desc="Number of hidden size for each hidden"
                 " layer"
        ))
        return params

    def build(self):
        """
        Build model structure.

        aNMM model based on bin weighting and query term attentions
        """
        # query is [batch_size, left_text_len]
        # doc is [batch_size, right_text_len, bin_num]
        query, doc, freq_vec, d_one_tensors, q_one_tensors = self._make_inputs()
        embedding = self._make_embedding_layer()

        q_embed0 = embedding(query)
        
        d_bin0 = tensorflow.keras.layers.Dropout(
            rate=self._params['dropout_rate'])(doc)
            
        #frequency vector
        
        d_bin1 = tensorflow.keras.layers.Dropout(
            rate=self._params['dropout_rate'])(freq_vec)
        
        #d_bin1 = tensorflow.cast(d_bin1, tensorflow.int32)
        
        #d_bin1 = tensorflow.keras.layers.Reshape((d_bin0,))(d_bin1)
            
        d_bin = tensorflow.keras.layers.Concatenate()([d_bin0, d_bin1])
        
        for layer_id in range(self._params['num_layers'] - 1):
            d_bin = tensorflow.keras.layers.Dense(
                self._params['hidden_sizes'][layer_id],
                kernel_initializer=RandomUniform())(d_bin)
            d_bin = tensorflow.keras.layers.Activation('tanh')(d_bin)
        
        d_bin = tensorflow.keras.layers.Dense(
            self._params['hidden_sizes'][self._params['num_layers'] - 1])(
            d_bin)
        
        # Score 1
        score0 = tensorflow.keras.layers.Dot(axes=[1, 1])([d_bin, d_one_tensors])
        x_out0 = self._make_output_layer()(score0)
        '''
        pos_vec = x_out0
        x_out0 = tensorflow.sort(x_out0, axis=1, direction='DESCENDING', name=None)
        
        piter=0
        for positionv in pos_vec:
            pind = tensorflow.where([], x=None, y=None, name=None)
        				pind = x_out0.index(positionv)
        				pos_vec[piter] = int(1/float(pind))
        				piter = piter + 1
        
        pos_vec = embedding(pos_vec)
        q_embed1 = tensorflow.keras.layers.Dropout(
            rate=self._params['dropout_rate'])(pos_vec)
        
        #q_embed1 = tensorflow.keras.layers.Reshape((q_embed0,))(q_embed1)
        '''
        q_embed = tensorflow.keras.layers.Concatenate()([q_embed0, x_out0])
        
        q_attention = tensorflow.keras.layers.Dense(
            1, kernel_initializer=RandomUniform(), use_bias=False)(q_embed)
        q_text_len = self._params['input_shapes'][0][0]
        q_text_len = 2 * q_text_len

        q_attention = tensorflow.keras.layers.Lambda(
            lambda x: softmax(x, axis=1),
            output_shape=(q_text_len,)
        )(q_attention)
        
        # Score 2
        score = tensorflow.keras.layers.Dot(axes=[1, 1])([q_attention, q_one_tensors])
        x_out = self._make_output_layer()(score)
        
        self._backend = keras.Model(inputs=[query, doc], outputs=x_out)
