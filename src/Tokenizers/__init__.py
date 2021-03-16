from src.Tokenizers.thread_tokenizer import Thread_Tokenizer
from transformers import RobertaTokenizer

def update_config(lm_tokeniser, config):
    print("Vocabulary : ", lm_tokeniser.tokenizer.get_vocab())

    config['vocab_size'] = lm_tokeniser.tokenizer.get_vocab_size()

    #Tokenization ids  
    config['mask_id'] = lm_tokeniser.tokenizer.token_to_id("<mask>")
    config['pad_id'] = lm_tokeniser.tokenizer.token_to_id("<pad>")
    config['sos_id'] = lm_tokeniser.tokenizer.token_to_id("<s>")
    config['eos_id'] = lm_tokeniser.tokenizer.token_to_id("</s>")
    config['dsm_list'] = [lm_tokeniser.tokenizer.token_to_id(token)
                                for token in lm_tokeniser.dms]
    config['total_steps'] = len([0 for thread in train_data_loader.thread_generator()])
    print("Total steps: ", config['total_steps'])

    return config

def get_tokenizer(config):
    config['pt_hf_tokenizer'] = RobertaTokenizer.from_pretrained('distilroberta-base')
    lm_tokeniser = Thread_Tokenizer(config)
    return lm_tokeniser, update_config(lm_tokeniser, config)
