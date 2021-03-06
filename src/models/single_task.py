import jax
import numpy as np
import jax.numpy as jnp
import haiku as hk

from src.models.crf import crf_layer
from src.models.embeddings import Embedding

class BiLSTM(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.forward_lstm = hk.DeepRNN([hk.LSTM(self.config['d_model']//2) for i in range(self.config['n_layers'])])
        self.backward_lstm = hk.DeepRNN([hk.LSTM(self.config['d_model']//2) for i in range(self.config['n_layers'])])
    
    def __call__(self, input_ids):
        embds = Embedding(self.config)(input_ids)

        state, forward_embds = hk.scan(lambda prev_state, inputs: self.forward_lstm(inputs, prev_state)[::-1],
                                       init = self.forward_lstm.initial_state(input_ids.shape[0]), 
                                       xs = jnp.transpose(embds, axes=(1,0,2)))
        
        state, backward_embds = hk.scan(lambda prev_state, inputs: self.backward_lstm(inputs, prev_state)[::-1],
                                        init = self.backward_lstm.initial_state(input_ids.shape[0]), 
                                        xs = jnp.transpose(embds, axes=(1,0,2)),
                                        reverse=True)
        
        return jnp.transpose(jnp.concatenate([forward_embds, backward_embds], axis=-1), axes=(1,0,2))

def get_loss_predict(config, key=jax.random.PRNGKey(42,)):

    def model_fn(token_ids, labels):
        crf = crf_layer(n_classes=len(config['class_names']), 
                        transition_init=hk.initializers.Constant(config['transition_init']),
                        scale_factors=config['scale_factors'],
                        init_alphas=config['init_alphas'])
        
        emmision_logits = hk.Linear(len(config['class_names']))(BiLSTM(config)(token_ids))
        return crf(emmision_logits, jnp.sum(token_ids!=config['pad_id'], axis=-1), labels)
    
    transformed_model_fn = hk.transform(model_fn)
    key, subkey = jax.random.split(key)
    model_params = transformed_model_fn.init(subkey, token_ids = np.random.randint(config['vocab_size'], size=(config['batch_size'], config['max_length'])),
                                             labels = np.random.randint(len(config['class_names']), size=(config['batch_size'], config['max_length'])))
  
    loss_fn = jax.jit(lambda params, key, token_ids, labels : transformed_model_fn.apply(params, key, token_ids, labels))
    
    def model(token_ids):
        crf = crf_layer(n_classes=len(config['class_names']), 
                        transition_init=hk.initializers.Constant(config['transition_init']),
                        scale_factors=config['scale_factors'],
                        init_alphas=config['init_alphas'])
        
        emmision_logits = hk.Linear(len(config['class_names']))(BiLSTM(config)(token_ids))
        return crf.batch_viterbi_decode(emmision_logits, lengths=jnp.sum(token_ids!=config['pad_id'], axis=-1),)[0]
    
    transformed_model = hk.transform(model)
    predict_fn = jax.jit(lambda params, key, token_ids : transformed_model.apply(params, key, token_ids))

    return model_params, loss_fn, predict_fn