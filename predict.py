import torch

import numpy as np
from nmt.models.seq2seq import Seq2Seq
from nmt.data_loader import DataLoader
from utils import preprocessing
import configparser
import translate

config = configparser.ConfigParser()
config.read('config/config.ini', encoding = 'utf-8')

saved_data = torch.load(
    config['system']['seq2seq_model'],
    map_location='cpu' if int(config['system']['gpu_id']) < 0 else 'cuda:%d'%int(config['system']['gpu_id'])
)

model_config = saved_data['config']
src_vocab, tgt_vocab, is_reverse = translate.get_vocabs(model_config, config, saved_data)

loader = DataLoader()
loader.load_vocab(src_vocab, tgt_vocab)

input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)

model = translate.get_model(input_size, output_size, model_config, is_reverse = is_reverse, saved_data=saved_data)
model.eval()

def translation(text):

    with torch.no_grad():
        # print(text)

        x = ([[word for word in text.split(' ')]], [len(text.split(' '))])
        #x_length = torch.tensor([len(x[0])])

        # print(loader.src.pad(x))
        x = loader.src.numericalize(
                x,
                device = 'cuda:%d' % int(config['system']['gpu_id']) if int(config['system']['gpu_id']) >= 0 else 'cpu'
            )
        
        # print(x)
        if int(config['system']['gpu_id']) >= 0:
            model.cuda(int(config['system']['gpu_id']))
    #        x = x.to('cuda:%d'%int(config['system']['gpu_id']))
    
    
        y_hat, indice = model.search(x)
    
        result = translate.to_text(indice, loader.tgt.vocab)

        return result[0]