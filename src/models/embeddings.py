import haiku as hk
import jax.numpy as jnp
import jax
import numpy as np

class Embedding(hk.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_emb_layer = hk.Embed(vocab_size=config['vocab_size'],
                                           embed_dim=config['d_model'])
        
    def __call__(self, token_ids, lang_ids=None):
        """
        token_ids: ints of shape (batch, n_seq)
        """
        
        flat_token_ids = jnp.reshape(token_ids, [-1])
        
        flat_token_embeddings = self.word_emb_layer(flat_token_ids)

        token_embeddings = jnp.reshape(flat_token_embeddings, [token_ids.shape[0], -1, self.config['d_model']])
        
        embeddings = token_embeddings + PositionEmbeddings(self.config)()[:token_embeddings.shape[1], :]
        
        if lang_ids is not None:
            embeddings += LanguageEmbeddings(self.config)(lang_ids)
        
        embeddings = hk.LayerNorm(axis=-1,
                                  create_scale=True,
                                  create_offset=True,)(embeddings)

        return embeddings


class PositionEmbeddings(hk.Module):
    """
    A position embedding of size [max_seq_leq, word_embedding_dim]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.offset = 2 if 'roberta' in self.config['initialize_pretrained'] else 0

    def get_init_pe(self):
        
        pe = np.zeros([self.config['max_length']+self.offset, self.config['d_model']])
        
        position = np.arange(0, self.config['max_length']+self.offset).reshape(-1,1)
        
        div_term = np.exp(np.arange(0, self.config['d_model'],2)*
                          -np.log(10000.0)/self.config['d_model'])
        
        pe[:, 0::2] = np.sin(position*div_term)
        pe[:, 1::2] = np.cos(position*div_term)
        
        return pe

    def __call__(self):
        
        position_weights = hk.get_parameter("position_embeddings",
                                            [self.config['max_length']+self.offset, self.config['d_model']],
                                            init=hk.initializers.Constant(self.get_init_pe()))
        
        start = self.offset
        end = self.offset+self.config['max_length']
        
        return position_weights[start:end]


class LanguageEmbeddings(hk.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, lang_ids):

        return hk.Embed(vocab_size=len(self.config['lang2id'])+1, 
                        embed_dim=self.config['d_model'])(lang_ids)