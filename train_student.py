from email import utils
import torch
import time
from engine import trainer, record_metric
import numpy as np
import utils.util as util
from data.DataHandler_st import DataHandler
from Yaml2Params import args, logger
import os
import random
# import tensorflow as tf

seed = 777

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # tf.random.set_seed(seed)
    torch.backends.cudnn.enabled = True



def train():
    set_seed(seed)
    
    handler = DataHandler()
    if args.dataset == 'PEMS7':
        tra_loader, val_loader, tst_loader, scaler, sp_adj, sp_adj_w= handler.get_dataloader(normalizer = args.norm)
    elif args.dataset == 'CHI_Cri': 
        tra_loader, val_loader, tst_loader, scaler, sp_adj, sp_adj_w= handler.get_dataloader_chi(normalizer = args.norm)
    elif args.dataset == 'wth2k_hum': 
        tra_loader, val_loader, tst_loader, scaler, sp_adj, sp_adj_w= handler.get_dataloader_hum(normalizer = args.norm)
    elif args.dataset in ['PEMS4S', 'PEMS7S', 'PEMS8S']: 
        tra_loader, val_loader, tst_loader, scaler, sp_adj, sp_adj_w= handler.get_dataloader_small(normalizer = args.norm)
    else:
        raise ValueError(f'{args.dataset} is not defined')
    # if args.model == 'gwnet': 
    if args.t_model == 'stgcn': 
        adj_mx = util.process_adj(sp_adj, args.adj_type)
        supports = torch.Tensor(adj_mx[0])
    elif args.t_model in ['mtgnn', 'stemgnn', 'dmstgcn']: # not need to process adj
        supports = None
    else: 
        raise ValueError('Model :{} error in processing adj'.format(args.model))
    
    engine = trainer(scaler, sp_adj, sp_adj_w, supports = supports)
    if args.load_t_out: 
        engine.load_t_model_out(tra_loader)
    tra_val_metric = dict()
    if args.testonly is not True:
        logger.info('start training .....')
        for epoch in range(1, args.max_epoch+1):
            print('*'*20, 'Training Process', '*'*20)
            t1 = time.time()
            tra_val_metric= engine.train_s(epoch, tra_loader, tra_val_metric)
            t2 = time.time()
            print('*'*20, 'Validating Process', '*'*20)
            tra_val_metric, stopFlg = engine.validation(epoch, val_loader, tra_val_metric)
            tra_val_metric = record_metric(tra_val_metric, [t2 - t1], ['cost time'])
            logger.info(tra_val_metric)
            if stopFlg:
                break
        logger.info('start testing .....')
        engine.test(tst_loader)        

    else:
        logger.info('start testing .....')
        engine.test(tst_loader)   


if __name__ == "__main__":
    t1 = time.time()
    train()

    t2 = time.time()
    logger.info("Total time spent: {:.4f}".format(t2 - t1))

