import torch
import torch.nn as nn
from models.networks.transformer_networks.clip.clip import tokenize
from utils.util import instantiate_from_config


class BaseCodec(nn.Module):
    
    def get_tokens(self, x, **kwargs):
        """
        Input: 
            x: input data
        Return:
            indices: B x L, the codebook indices, where L is the length 
                    of flattened feature map size
        """
        raise NotImplementedError

    def get_number_of_tokens(self):
        """
        Return: int, the number of tokens
        """
        raise NotImplementedError

    def encode(self, img):
        raise NotImplementedError

    def decode(self, img_seq):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        if self.trainable and mode:
            return super().train(True)
        else:
            return super().train(False)

    def _set_trainable(self):
        if not self.trainable:
            for pn, p in self.named_parameters():
                p.requires_grad = False
            self.eval()


class Tokenize(BaseCodec):
    def __init__(self, context_length:int = 256,
                 add_start_and_end:bool = False,
                 just_token = False,
                 with_mask:bool = True,
                 pad_value:int = 0,
                 clip_embedding = False,
                 condition_emb_config = None,
                 tokenizer_config={
                     'target': 'image_synthesis.modeling.modules.clip.simple_tokenizer.SimpleTokenizer',
                     'params':{
                        'end_idx': 49152 # 16384 fo DALL-E
                        },
                 },
                 ):
        """
        This is a wrapper class for tokenize of texts.
        For CLIP and DALLE-pytorch tokenize, the default
        arguments are different:

        CLIP based:
            context_length: 77
            add_start_and_end: True

        DALLE-pytorch based:
            context_length: 256
            add_start_and_end: False
        
        """
        super().__init__()
        self.context_length = context_length
        self.add_start_and_end = add_start_and_end
        self.with_mask = with_mask
        self.pad_value = pad_value
        self.just_token = just_token
        self.trainable = False
        self.condition_emb = None
        self.clip_embedding = clip_embedding
        if self.clip_embedding == True:
            assert condition_emb_config != None
            self.condition_emb = instantiate_from_config(condition_emb_config)

        self.tokenizer = instantiate_from_config(tokenizer_config)
    
    def __repr__(self):
        rep = "Tokenize for text\n\tcontent_length: {}\n\tadd_start_and_end: {}\n\twith_mask: {}"\
                .format(self.context_length, self.add_start_and_end, self.with_mask)
        return rep

    def check_length(self, token):
        return len(token) <= self.context_length

    def get_tokens(self, text, **kwargs):
        text_token = tokenize(text, context_length=self.context_length, 
                         add_start_and_end=self.add_start_and_end,
                         with_mask=self.with_mask, pad_value=self.pad_value,
                         tokenizer=self.tokenizer,
                         just_token=self.just_token)
        if self.clip_embedding == False:
            return text_token
        else:
            if self.condition_emb.additional_last_embedding == True:
                with torch.no_grad():
                    cond_emb, last_embedding = self.condition_emb(text_token['token'].cuda()) 
                    text_token['embed_token'] = cond_emb.detach()
                    text_token['last_embed'] = last_embedding
            else:
                with torch.no_grad():
                    cond_emb = self.condition_emb(text_token['token'].cuda())
                    text_token['embed_token'] = cond_emb.detach()

            return text_token
