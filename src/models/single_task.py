import jax
import numpy as np
import jax.numpy as jnp
import haiku as hk

from src.models.crf import crf_layer
from src.models.embeddings import Embedding

class BiLSTM(hk.Module):
    def __init__(self, config):
        self.config = config

    def __call__(self, input_ids):
        embds = Embedding(self.config)(input_ids)
        lstm = hk.LSTM(self.config['d_model']//2)
        
        state, forward_embds = jax.lax.scan(lambda prev_state, inputs: lstm(inputs, prev_state)[::-1],
                                            init = jnp.ones_like(embds[:,0,:]), 
                                            xs = jnp.transpose(embds, axes=(0,1)))
        
        state, backward_embds = jax.lax.scan(lambda prev_state, inputs: lstm(inputs, prev_state)[::-1],
                                            init = jnp.ones_like(embds[:,0,:]), 
                                            xs = jnp.transpose(embds, axes=(0,1)),\
                                            reverse=True)
        
        return jnp.transpose(jnp.concat(forward_embds, backward_embds), axes=(0,1))

def get_loss_predict(config, key=jax.random.PRNGKey(42,)):

    def model(token_ids, labels):
        crf = crf_layer(n_classes=len(config['class_names']), 
                        transition_init=config['transition_init'],
                        scale_factors=config['scale_factors'],
                        init_alphas=config['init_alphas'])
        
        emmision_logits = BiLSTM(config)(token_ids)
        return crf(emmision_logits, jnp.sum(token_ids!=config['pad_id'], axis=-1), labels)
    
    transformed_model = hk.transform(model)
    key, subkey = jax.random.split(key)
    model_params = transformed_model.init(subkey, token_ids = np.random.randint(config['vocab_size'], size=(config['batch_size'], config['max_length'])),
                                          labels = np.random.randint(len(config['class_names']), size=(config['batch_size'], config['max_length']))))
    
    loss_fn = jax.jit(lambda params, key, token_ids, labels : transformed_model.apply(params, key, token_ids, labels))

    def model(token_ids):
        crf = crf_layer(n_classes=len(config['class_names']), 
                        transition_init=config['transition_init'],
                        scale_factors=config['scale_factors'],
                        init_alphas=config['init_alphas'])
        
        emmision_logits = BiLSTM(config)(token_ids)
        return crf.batch_viterbi_decode(emmision_logits, lengths=jnp.sum(token_ids!=config['pad_id'], axis=-1),)[0]
    
    transformed_model = hk.transform(model)
    predict_fn = jax.jit(lambda params, key, token_ids : transformed_model.apply(params, key, token_ids))

    return model_params, loss_fn, predict_fn