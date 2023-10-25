import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=0)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class Encoder(nn.Module):
    """ An encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, device):
        super().__init__()

        self.d_model = d_model
        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=True)
            for _ in range(n_layers)])

    def temporal_enc(self, event_type, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result =  (torch.arange(event_type.shape[1]).to('cuda')+ torch.zeros_like(event_type)).unsqueeze(-1)  / self.position_vec

        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, relation_mat, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        

        tem_enc = self.temporal_enc(event_type, non_pad_mask)
        enc_output = self.event_emb(event_type)
#         enc_output =  torch.zeros_like(tem_enc)


        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, attn_weights = enc_layer(
                enc_output,  relation_mat,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output, attn_weights

    
# class Decoder(nn.Module):
#     """ A decoder model with relation-aware attention mechanism. """

#     def __init__(
#             self,
#             num_types, d_model, d_inner,
#             n_layers, n_head,  dropout, device):
#         super().__init__()

#         self.d_model = d_model
#         self.num_types = num_types
#         self.n_head = n_head
#         self.device = device

#         # position vector, used for temporal encoding
#         self.position_vec = torch.tensor(
#             [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
#             device=device)

#         # event type embedding
#         self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

#         self.layer_stack = nn.ModuleList([
#             DecoderLayer(d_model, d_inner, n_head, dropout=dropout, normalize_before=False)
#             for _ in range(n_layers)])
      
        

#     def temporal_enc(self, event_type, non_pad_mask):
#         """
#         Input: batch*seq_len.
#         Output: batch*seq_len*d_model.
#         """

#         result =  (torch.arange(event_type.shape[1])+ torch.zeros_like(event_type)).unsqueeze(-1)  / self.position_vec
#         result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
#         result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
#         return result * non_pad_mask
    
#     def attention_from_relation(self, relation, event_type, num_types):
#         batchlab = F.one_hot(event_type, num_classes = num_types+1).type(torch.FloatTensor)
#         intermediate = torch.matmul(batchlab[:,:,1:],relation)
#         attention = torch.matmul(intermediate,batchlab[:,:,1:].permute(0,2,1))
#         # repeat attention along the dimension of nheads
#         attention_indicator = attention.unsqueeze(1).repeat(1, self.n_head, 1,1).to(self.device)
#         return attention_indicator 

#     def forward(self, event_type, non_pad_mask, relation_matrix):
#         """ Encode event sequences via masked self-attention. """

#         # prepare attention masks
#         # slf_attn_mask is where we cannot look, i.e., the future and the padding
#         slf_attn_mask_subseq = get_subsequent_mask(event_type)
#         slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
#         slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
#         slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
#         slf_attn_mask = slf_attn_mask.unsqueeze(1).repeat(1, self.n_head, 1,1)
        
        

#         tem_enc = self.temporal_enc(event_type, non_pad_mask)
#         dec_output = self.event_emb(event_type)

#         relation_indicator = self.attention_from_relation(relation_matrix, event_type, self.num_types)
        
#         for dec_layer in self.layer_stack:
#             dec_output += tem_enc
#             dec_output = dec_layer(
#                 dec_output, relation_indicator,
#                 non_pad_mask=non_pad_mask,
#                 slf_attn_mask=slf_attn_mask)
          
#         return dec_output
    



class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_inner=256,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, device = 'cuda'):
        super().__init__()
  
        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device = device
        )
    
    
        
#         self.decoder = Decoder(
#             num_types=num_types,
#             d_model=d_model,
#             d_inner=d_inner,
#             n_layers=n_layers,
#             n_head=n_head,
#             dropout=dropout,
#             device = device
#         )

        self.num_types = num_types
        self.device = device
        self.n_head = n_head

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)
        self.linear2 = nn.Linear(num_types, num_types,bias=True)
   
        

    def attention_from_relation(self, relation, event_type, num_types):
        batchlab = F.one_hot(event_type, num_classes = num_types+1).type(torch.FloatTensor).to('cuda')
        intermediate = torch.matmul(batchlab[:,:,1:],relation.to('cuda'))
        attention = torch.matmul(intermediate,batchlab[:,:,1:].permute(0,2,1))
        # repeat attention along the dimension of nheads
        attention_indicator = attention.unsqueeze(1).repeat(1, self.n_head, 1,1).to('cuda')
        return attention_indicator       


#     def relation_from_attention(self, attn_weights, event_type, num_types):
#         # attn_weights :batch x n_heads x seq_length x seq_length 
#         #ave_attn = torch.mean(attn_weights,dim=1)
        
#         # batchlab :batch x seq_length x num_class 
#         batchlab = F.one_hot(event_type, num_classes = num_types+1).type(torch.FloatTensor).to(self.device)
#         intermediate = torch.matmul(batchlab.permute(0,2,1).unsqueeze(1).repeat(1, attn_weights.shape[1], 1,1) ,attn_weights)        
#         relation = torch.matmul(intermediate,batchlab.unsqueeze(1).repeat(1, attn_weights.shape[1], 1,1))
   
#         return torch.mean(relation,dim=(0,1))[1:,1:].T

    def relation_from_attention(self, attn_weights, event_type, event_time, num_types, decay_rate):
        # attn_weights :batch x n_heads x seq_length x seq_length 
        time_diff = event_time.unsqueeze(-1).repeat(1,1, len(event_time[0])) -  event_time.unsqueeze(1)
        masked_time_diff = time_diff * (time_diff>0)
        masked_time_diff = torch.tril(masked_time_diff,diagonal=-1) 
        non_pad_mask = get_non_pad_mask(event_type)
        temporal_decay = torch.tril(torch.exp(- decay_rate * masked_time_diff  ) * non_pad_mask, diagonal=-1)
       # print(attn_weights[0,0,:,:])
        decay_attn =  attn_weights * temporal_decay.unsqueeze(1).repeat(1, attn_weights.shape[1], 1,1)
        
        # batchlab :batch x seq_length x num_class 
        batchlab = F.one_hot(event_type, num_classes = num_types+1).type(torch.FloatTensor).to(self.device)
        intermediate = torch.matmul(batchlab.permute(0,2,1).unsqueeze(1).repeat(1, attn_weights.shape[1], 1,1) , decay_attn)        
        relation = torch.matmul(intermediate,batchlab.unsqueeze(1).repeat(1, attn_weights.shape[1], 1,1))
   
        return torch.mean(relation,dim=(0,1))[1:,1:].T
    
#     def sampling(self, num_samples, binarylogits):
#         samples = torch.zeros((num_samples, self.num_types))
#         for i in range(num_samples):
#            # print( F.gumbel_softmax(binarylogits, tau=1, hard=True).shape)
#             samples[i,:] = F.gumbel_softmax(binarylogits, tau=1, hard=True)[:,1]
#         return samples
    def sampling(self, num_samples, binarylogits):
        samples = torch.zeros((num_samples,self.num_types,self.num_types))
        for i in range(num_samples):
            samples[i,:,:] = F.gumbel_softmax(binarylogits, tau=1, hard=True)[:,:,1]
        return samples
    
    def forward(self, event_type, event_time, num_samples, decay_rate ):
        """
        Return the hidden representations from decoder.
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: output: batch*seq_len*model_dim;
                binrel: binarized relation matrix P(A_ij=0), P(A_ij=1)  (num_typesxnum_types) x 2 
                relation matrix: encoding P(A_ij=1) for each entry A_ij  (num_typesxnum_types)
               
        """
        #encoding step
        non_pad_mask = get_non_pad_mask(event_type)
        # only output attention weights from encoder: batch x n_heads x seq_len x seq_len
        output, attn_weights = self.encoder(event_type, torch.ones((event_type.shape[0],self.n_head,event_type.shape[1],event_type.shape[1])).to('cuda'), non_pad_mask)
        # summarize attention to relation; K x K (K : num_types)
        relation = self.relation_from_attention(attn_weights, event_type, event_time, self.num_types, decay_rate )
        
#         print(relation)
        
        # map relations to a score between [0,1] to denote P(A_ij=1) 
        relation_output = torch.sigmoid(self.linear2(relation))
        
#         print(relation_output)
        #sampling: sample n samples from gumbel-softmax via sampling function
        # output a 2d matrix binary relation (binrel) of (KxK) x 2 : a vectorized P(A_ij=0), P(A_ij=1)
       
        rel = torch.flatten(relation_output) 
        binrel = torch.stack([1-rel,rel])

        binaryProb =  binrel.reshape(2,self.num_types,self.num_types)
        binarylogits =  torch.log(binaryProb.permute(1,2,0)+1e-15)  
        samples = self.sampling(num_samples,binarylogits)

        
        # decoder: with relation-aware attention, output from attention  
        output = torch.tensor(()).to(self.device)
        
        for i in range(num_samples):
            dec_output,_= self.encoder(event_type, self.attention_from_relation( samples[i], event_type, self.num_types), non_pad_mask)
            output = torch.cat((output, dec_output), 0)
        output = output.view(num_samples,dec_output.shape[0],dec_output.shape[1],dec_output.shape[2])
        

        return output , binrel, relation_output

