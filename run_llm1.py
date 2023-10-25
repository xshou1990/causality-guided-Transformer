### incorporate impeachment -> protest ###s


import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import sklearn
import sklearn.metrics

import transformer.Constants as Constants
# import Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm
# import seaborn as sns
    


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    devloader = get_dataloader(dev_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, devloader, testloader, num_types





def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    num_iter = 0 # number of batches per epoch
    
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):

        num_iter += 1
        """ prepare data """
        _,_, event_type = map(lambda x: x.to(opt.device), batch)
        
        event_type_0 = torch.hstack([torch.zeros(event_type.shape[0],1).int().to('cuda'),event_type])

        """ forward """
        optimizer.zero_grad()

        output, _ = model(event_type_0)
        
        loss = -log_likelihood(model, output[:,:-1,:], event_type )

        z_score, y_score = zyscores(model, output[:,:-1,:], event_type, opt.type_z, opt.type_y )
        # B excites A 
        if opt.sle == 0 and opt.excites == 1: # we need penalize effect or non excites if nonexcites produces large probability for every instance in a sequence
          loss_cp =  F.relu((event_type!=opt.type_z)[:,:-1]* y_score[:,1:]/(1.0-z_score[:,:-1])- (event_type==opt.type_z)[:,:-1] * y_score[:,1:]/z_score[:,:-1]) 
          loss_cp = torch.sum(loss_cp)
        elif opt.sle == 0 and opt.excites == 0:
          loss_cp =  F.relu( (event_type==opt.type_z)[:,:-1] * y_score[:,1:]/z_score[:,:-1] - (event_type!=opt.type_z)[:,:-1]* y_score[:,1:]/(1.0-z_score[:,:-1]) ) 
          loss_cp = torch.sum(loss_cp)
        elif opt.sle == 1 and opt.excites == 1:
          loss_cp =  torch.mean((event_type!=opt.type_z)[:,:-1]* y_score[:,1:]/(1.0-z_score[:,:-1])- (event_type==opt.type_z)[:,:-1] * y_score[:,1:]/z_score[:,:-1], dim=-1) 
          loss_cp = torch.sum(F.relu(loss_cp))
        else:
          loss_cp =  torch.mean( (event_type==opt.type_z)[:,:-1] * y_score[:,1:]/z_score[:,:-1]- (event_type!=opt.type_z)[:,:-1]* y_score[:,1:]/(1.0-z_score[:,:-1]), dim=-1) 
          loss_cp = torch.sum(F.relu(loss_cp))
        
        
        if opt.alpha != 0:
          print(loss_cp)
          loss += opt.alpha * loss_cp
                
        """ backward """
        loss.backward()

        """ update parameters """
        optimizer.step()
    

    return -loss


def eval_epoch(model, validation_data, opt, event_interest=None):
    """ Epoch operation in evaluation phase. """

    model.eval()
    
    total_event_ll =0 # total loglikelihood
    num_iter = 0
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):

            num_iter +=1
#             print(num_iter)
            """ prepare data """
            _,_, event_type = map(lambda x: x.to(opt.device), batch)
            
            event_type_0 = torch.hstack([torch.zeros(event_type.shape[0],1).int().to('cuda'),event_type])

            """ forward """
            
            output, _ = model(event_type_0 )
            
            #average_attn = model.relation_from_attention( attn_weights, event_type_0, model.num_types)
            
            """ compute loss """
            # negative log-likelihood conditioned on relation sample
            if event_interest is None:
              event_ll = log_likelihood(model, output[:,:-1,:], event_type)
            else:
              event_ll = log_likelihood_event(model, output[:,:-1,:], event_type, event_interest)
            
            total_event_ll +=  event_ll
            

    return  total_event_ll #, average_attn[event_interest-1,:]/torch.max(average_attn[event_interest-1,:])


def train(model, training_data, validation_data, test_data, optimizer, scheduler, opt, event_interest):
    """ Start training. """
    
    best_ll = -np.inf
    best_model = deepcopy(model.state_dict())

    train_ll_list = [] # train log likelihood
    valid_ll_list = [] # valid log likelihood
    impatience = 0 
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_ll = train_epoch(model, training_data, optimizer, opt )
        
        train_ll_list +=[train_ll]
        
        print('  - (Training)     loglikelihood: {ll: 8.4f} ,'
              'elapse: {elapse:3.3f} min'
              .format( ll=train_ll, elapse=(time.time() - start) / 60))
        
    
        start = time.time()
        
        valid_ll = eval_epoch(model, validation_data, opt, event_interest )
        valid_ll_list += [valid_ll]
        print('  - (validation)  loglikelihood: {ll: 8.4f}'
              'elapse: {elapse:3.3f} min'
              .format( ll= valid_ll, elapse=(time.time() - start) / 60))

        start = time.time()
        
        test_ll = eval_epoch(model, test_data, opt, event_interest )
        print('  - (test)  loglikelihood: {ll: 8.4f}'
              'elapse: {elapse:3.3f} min'
              .format( ll= test_ll, elapse=(time.time() - start) / 60))
        
        print('  - [Info] Maximum validation loglikelihood:{ll: 8.4f} '
              .format(ll = max(valid_ll_list) ))
        

        if (valid_ll- best_ll ) < 1e-3:
            impatient += 1
            if best_ll < valid_ll:
                best_ll = valid_ll
                best_model = deepcopy(model.state_dict())
        else:
            best_ll = valid_ll
            best_model = deepcopy(model.state_dict())
            impatient = 0
        
            
        if impatient >= 5:
            print(f'Breaking due to early stopping at epoch {epoch}')
            break
    

        
        scheduler.step()
    

    return best_model
        
def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def zyscores(model, data, types, type_z, type_y):
    """ penality term """

    all_hid = model.linear(data)[:,:,1:]
    
    all_scores = F.softmax(all_hid,dim=-1)

    type_z_scores = all_scores[:,:,type_z-1]

    type_y_scores = all_scores[:,:,type_y-1]

    return type_z_scores, type_y_scores

def log_likelihood(model, data, types):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    all_hid = model.linear(data)[:,:,1:]
    
    all_scores = F.log_softmax(all_hid,dim=-1)

   # print( type_z_scores,  type_y_scores )

    # print("hello")
    # print(all_scores.shape)
    types_3d =  F.one_hot(types, num_classes= model.num_types+1)
  
    ll = (all_scores*types_3d[:,:,1:]) #[:,1:,:]
    
    ll2 = torch.sum(ll,dim=-1)*non_pad_mask
    ll3 = torch.mean(torch.sum(ll2,dim=-1))



    return ll3



def log_likelihood_event(model, data, types, event_interest):
    """ Log-likelihood of observing event of interest in the sequence. """


    non_pad_mask = get_non_pad_mask(types).squeeze(2)
   
    all_hid = model.linear(data)

    all_scores = F.softmax(all_hid,dim=-1)
    all_scores_event = torch.log(all_scores[:,:,event_interest-1] +1e-12)
    all_scores_nonevent = torch.log(1 - all_scores[:,:,event_interest-1] +1e-12 )

    event_log_ll = (types == event_interest) * all_scores_event
    nonevent_log_ll = (types != event_interest) * all_scores_nonevent
    ll = (event_log_ll + nonevent_log_ll)*non_pad_mask#[:,1:]
    ll2 = torch.sum(ll)

    return ll2
    

import sys
sys.argv=['']
del sys



# for seed_i in range(4,5):
#     print("current simulation is {}".format(seed_i))
#     np.random.seed(seed_i)
    
#     #generate data
#     num_seqs = 50
#     num_events_per_seq = 100
#     synthetic_data = gen_mult_seq_example(num_seqs, num_events_per_seq)
    
#     # preprocess
#     dic = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
#     dict_list_total = []
#     for i, (k, v) in enumerate(synthetic_data.items()):
#         event_times = []
#         event_types = []
#         dict_list = []
#         counter = 1
#         for j in v:
#             event_times.append( counter )
#             event_types.append(dic[j])
#             counter += 1
#         inter_times = np.ediff1d(event_times)
#         inter_times = np.insert(inter_times, 0,0)
#         for l in range(len(event_times)):
#             dicti = {'time_since_start': event_times[l],
#             'time_since_last_event': inter_times[l],
#             'type_event': event_types[l] }
#             dict_list.append(dicti) 
#         dict_list_total.append(dict_list)
     
parsed_args = argparse.ArgumentParser()
parsed_args.device = 0
parsed_args.batch_size = 32
parsed_args.n_head = 4
parsed_args.n_layers = 4
parsed_args.d_model = 256
parsed_args.d_inner = 512
parsed_args.d_k=256
parsed_args.d_v=256
parsed_args.dropout=0.1
parsed_args.lr=4e-4   #2.5e-5
parsed_args.epoch=500
parsed_args.type_z = 3 #          (C, B) , C excites B
parsed_args.type_y = 2 #  
parsed_args.sle = 1 # sequence level effect ==> 1 or instance level effect ==> 0 
parsed_args.excites = 1 # excites effect ==> 1, inhibits effect ==> 0 
parsed_args.alpha = 10
parsed_args.data = "data/llm_data/"

opt = parsed_args

print('[Info] parameters: {}'.format(opt))

# default device is CUDA
#     opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# default device is CUDA temporary
opt.device = torch.device("cuda")

trainloader, devloader, testloader, num_types = prepare_dataloader(opt)

#     trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
#     devloader = get_dataloader(dev_data, opt.batch_size, shuffle=True)
#     testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)

# num_types = 50


torch.manual_seed(0)

""" prepare model """
model = Transformer(
    num_types=num_types,
    d_model=opt.d_model,
    d_inner=opt.d_inner,
    n_layers=opt.n_layers,
    n_head=opt.n_head,
    d_k=opt.d_k,
    d_v=opt.d_v,
    dropout=opt.dropout,
)
model.to(opt.device)

""" optimizer and scheduler """
optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                        opt.lr, betas=(0.9, 0.999), eps=1e-08)

scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9)


""" number of parameters """
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('[Info] Number of parameters: {}'.format(num_params))

_ = train(model, trainloader, devloader, testloader, optimizer, scheduler, opt, event_interest=None)

#     for event_interest in [0,1,2,3,4]:
#     #     """ train model"""
#         model.load_state_dict(best_model)
    
#         model.eval()
    
#         test_ll = eval_epoch(model, testloader, opt, event_interest)

#         print(" event_interest {}: test log likelihood {} ".format( event_interest, test_ll))
    
