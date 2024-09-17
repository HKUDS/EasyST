from scipy import io
import numpy as np
import datetime
from Yaml2Params import args, logger
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
import os
import os.path as osp
# from dgl.dataloading import GraphDataLoader
# from data.STGDataset import STGDataset
from utils.util import Add_Window_Horizon_time, normalize_dataset, split_data_by_ratio, Add_Window_Horizon,get_adjacency_binary, data_loader, pair_dist_to_adj, Grid2Graph, build_grid_adj_weight, LatLon_to_adj
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .dataset import SAINTDataset
import scipy.sparse as sp

class DataHandler:
    def __init__(self):
        # pass
        self.saint_batch_size = 500
        self.saint_sample_type = 'random_walk'
        self.saint_walk_length = 1
        self.saint_shuffle_order = 'node_first'
    
    def get_dataloader(self, normalizer='max01'):
        data, time, adj_dist, loc_list = self.load_st_dataset(args.dataset)  # B, N, D
        data, scaler = normalize_dataset(data, normalizer, False)
        time = self.build_time(time)
        x_tra, y_tra, x_val, y_val, x_test, y_test = self.get_raw_data(data)
        TEx_tra, TEy_tra, TEx_val, TEy_val, TEx_tst, TEy_tst = self.get_raw_data(time)
        tra_TE = np.concatenate([TEx_tra, TEy_tra], axis = 1)
        val_TE = np.concatenate([TEx_val, TEy_val], axis = 1)
        tst_TE = np.concatenate([TEx_tst, TEy_tst], axis = 1)

        adj, adj_weight = self.build_adj(adj_dist)
        self.init_graph(adj, adj_weight, loc_list)
        
        sp_adj = adj
        sp_adj_w = adj_weight
        eval_adj(sp_adj, 'sp_binary')
        eval_adj(sp_adj_w, 'sp_weight')
        # eval_adj(temp_adj, 'temp')
        if args.is_sample is False:
            tra_loader = data_loader(x_tra, y_tra, tra_TE, args.batch_size, shuffle=True, drop_last=True)
            val_loader = data_loader(x_val, y_val, val_TE, args.batch_size, shuffle=False, drop_last=True)
            tst_loader = data_loader(x_test, y_test, tst_TE, args.batch_size, shuffle=False, drop_last=False)
        else: 
            tra_loader = self.make_sample_dataloader([x_tra, y_tra, tra_TE], True)
            val_loader = self.make_sample_dataloader([x_val, y_val, val_TE], False)
            tst_loader = self.make_sample_dataloader([x_test, y_test, tst_TE], False)

        return tra_loader, val_loader, tst_loader, scaler, sp_adj, sp_adj_w
    def get_dataloader_small(self, normalizer='max01'):
        data, time, _, _ = self.load_st_dataset(args.dataset)  # B, N, D
        data, scaler = normalize_dataset(data, normalizer, False)
        time = self.build_time(time)
        x_tra, y_tra, x_val, y_val, x_test, y_test = self.get_raw_data(data)
        TEx_tra, TEy_tra, TEx_val, TEy_val, TEx_tst, TEy_tst = self.get_raw_data(time)
        tra_TE = np.concatenate([TEx_tra, TEy_tra], axis = 1)
        val_TE = np.concatenate([TEx_val, TEy_val], axis = 1)
        tst_TE = np.concatenate([TEx_tst, TEy_tst], axis = 1)

        adj, adj_weight = self.build_adj_small()

        sp_adj = adj
        sp_adj_w = adj_weight

        tra_loader = data_loader(x_tra, y_tra, tra_TE, args.batch_size, shuffle=True, drop_last=True)
        val_loader = data_loader(x_val, y_val, val_TE, args.batch_size, shuffle=False, drop_last=True)
        tst_loader = data_loader(x_test, y_test, tst_TE, args.batch_size, shuffle=False, drop_last=False)

        return tra_loader, val_loader, tst_loader, scaler, sp_adj, sp_adj_w
    def get_dataloader_chi(self, normalizer='max01'):
        data, time, adj, loc_list = self.load_st_dataset(args.dataset)  # B, N, D
        data, scaler = normalize_dataset(data, normalizer, False)
        time = self.build_time(time)
        x_tra, y_tra, x_val, y_val, x_test, y_test = self.get_raw_data(data)
        TEx_tra, TEy_tra, TEx_val, TEy_val, TEx_tst, TEy_tst = self.get_raw_data(time)
        tra_TE = np.concatenate([TEx_tra, TEy_tra], axis = 1)
        val_TE = np.concatenate([TEx_val, TEy_val], axis = 1)
        tst_TE = np.concatenate([TEx_tst, TEy_tst], axis = 1)

        adj_weight = build_grid_adj_weight(start_pos = (41.644585429, -87.939732936), end_pos = (42.022910333, -87.524529378), row = 42, col = 35, args=args)
        self.init_graph(adj, adj_weight, loc_list)
        
        sp_adj = adj
        sp_adj_w = adj_weight
        eval_adj(sp_adj, 'sp_binary')
        eval_adj(sp_adj_w, 'sp_weight')
        # eval_adj(temp_adj, 'temp')
        if args.is_sample is False:
            tra_loader = data_loader(x_tra, y_tra, tra_TE, args.batch_size, shuffle=True, drop_last=True)
            val_loader = data_loader(x_val, y_val, val_TE, args.batch_size, shuffle=False, drop_last=True)
            tst_loader = data_loader(x_test, y_test, tst_TE, args.batch_size, shuffle=False, drop_last=False)
        else: 
            tra_loader = self.make_sample_dataloader([x_tra, y_tra, tra_TE], True)
            val_loader = self.make_sample_dataloader([x_val, y_val, val_TE], False)
            tst_loader = self.make_sample_dataloader([x_test, y_test, tst_TE], False)

        return tra_loader, val_loader, tst_loader, scaler, sp_adj, sp_adj_w
    def build_adj_small(self, ):
        adj_ori = get_adjacency_binary(distance_df_filename=args.adj_filename,
                                       num_of_vertices=args.num_nodes, id_filename=args.id_filename, args = args)
        adj_weight = get_adjacency_binary(distance_df_filename=args.adj_filename,
                                       num_of_vertices=args.num_nodes, id_filename=args.id_filename, type_='distance', self_loop=True, args = args)
        
        return adj_ori, adj_weight

    def get_dataloader_hum(self, normalizer='max01'):
        data, time, adj_dist, loc_list = self.load_st_dataset(args.dataset)  # B, N, D
        data, scaler = normalize_dataset(data, normalizer, False)
        time = self.build_time(time)
        x_tra, y_tra, x_val, y_val, x_test, y_test = self.get_raw_data(data)
        TEx_tra, TEy_tra, TEx_val, TEy_val, TEx_tst, TEy_tst = self.get_raw_data(time)
        tra_TE = np.concatenate([TEx_tra, TEy_tra], axis = 1)
        val_TE = np.concatenate([TEx_val, TEy_val], axis = 1)
        tst_TE = np.concatenate([TEx_tst, TEy_tst], axis = 1)

        adj, adj_weight = self.build_adj(adj_dist)
        self.init_graph(adj, adj_weight, loc_list)
        
        sp_adj = adj
        sp_adj_w = adj_weight
        eval_adj(sp_adj, 'sp_binary')
        eval_adj(sp_adj_w, 'sp_weight')
        # eval_adj(temp_adj, 'temp')
        if args.is_sample is False:
            tra_loader = data_loader(x_tra, y_tra, tra_TE, args.batch_size, shuffle=True, drop_last=True)
            val_loader = data_loader(x_val, y_val, val_TE, args.batch_size, shuffle=False, drop_last=True)
            tst_loader = data_loader(x_test, y_test, tst_TE, args.batch_size, shuffle=False, drop_last=False)
        else: 
            tra_loader = self.make_sample_dataloader([x_tra, y_tra, tra_TE], True)
            val_loader = self.make_sample_dataloader([x_val, y_val, val_TE], False)
            tst_loader = self.make_sample_dataloader([x_test, y_test, tst_TE], False)

        return tra_loader, val_loader, tst_loader, scaler, sp_adj, sp_adj_w
    def init_graph(self, adj, adj_weight, loc_list):
        adj_weight = sp.coo_matrix(adj_weight)

        edge_index, edge_weight = from_scipy_sparse_matrix(adj_weight)
        self.edge_index = torch.LongTensor(edge_index)
        self.edge_weight = torch.DoubleTensor(edge_weight)
               
        if loc_list is not None:
            self.node_name = loc_list
            assert args.num_nodes == len(self.node_name)

        self.num_edges = self.edge_weight.shape[0]

    def make_sample_dataloader(self, inputs, shuffle):
        
        dataset = SAINTDataset(
            inputs,
            self.edge_index, self.edge_weight, args.num_nodes,
            args.batch_size, shuffle=shuffle,
            shuffle_order=self.saint_shuffle_order,
            saint_sample_type=self.saint_sample_type,
            saint_batch_size=self.saint_batch_size,
            saint_walk_length=self.saint_walk_length,
        )

        return DataLoader(dataset, batch_size=None)
    
    
    def get_raw_data(self, data):
        
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
        # add time window
        
        x_tra, y_tra = Add_Window_Horizon(data_train, window=args.lag, horizon=args.horizon, single=False)
        x_val, y_val = Add_Window_Horizon(data_val, window=args.lag, horizon=args.horizon, single=False)
        x_test, y_test = Add_Window_Horizon(data_test, window=args.lag, horizon=args.horizon, single=False)
        
        
        print('Train: ', x_tra.shape, y_tra.shape)
        print('Val: ', x_val.shape, y_val.shape)
        print('Test: ', x_test.shape, y_test.shape)
        return x_tra, y_tra, x_val, y_val, x_test, y_test

    def build_time(self, time):
        if args.dataset in ['PEMS7', 'wth2k_hum', 'PEMS4S', 'PEMS7S', 'PEMS8S', ]:
            dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
            timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
                        // time.freq.delta.total_seconds()
            timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
            time = torch.cat((timeofday, dayofweek), -1)
        elif args.dataset in ['CHI_Cri']: 
            dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
            dayofyear = torch.reshape(torch.tensor(time.dayofyear), (-1, 1))
            dayofyear = dayofyear - 1
            time = torch.cat((dayofyear, dayofweek), -1)

        return time
    def is_increasing(self, loc_list):
        return all(x<y for x, y in zip(loc_list, loc_list[1:]))
    
    def load_st_dataset(self, dataset):
        # output B, N, D
        if dataset == 'PEMS7':
            data_path = os.path.join('./data/PEMS07/pems07_9_2_6m_52128x1481x1.npz')
            fdata = np.load(data_path)
            raw_data = fdata['data'] # flow, occupy, speed
            assert raw_data.shape == (52128, 1481, 3), 'shape error: {}'.format(raw_data.shape)
            data = raw_data[:, :, 0]
            adj = fdata['adj'] 
            assert adj.shape == (1481, 1481)
            loc_list = fdata['loc_list']
            assert loc_list.shape == (1481,) and self.is_increasing(loc_list)
            start_date = '2022-09-01 00:00:00'
            time = pd.date_range(start_date, periods=data.shape[0], freq = '5MIN')
            print(f'The last day {time[-1]}')
        elif dataset == 'CHI_Cri': 
            data_path = os.path.join('./data/CHI_Crime/chi_crime_42x35x7670x4.npz')
            fdata = np.load(data_path)
            raw_data = fdata['data'] # flow, occupy, speed
            assert raw_data.shape == (42, 35, 7670, 4), 'shape error: {}'.format(raw_data.shape)
            data = raw_data.reshape(42*35, 7670, 4).transpose(1, 0, 2)
            assert data.shape == (7670, 1470, 4), 'shape error: {}'.format(data.shape)
            g2g_hander = Grid2Graph(row = raw_data.shape[0], col = raw_data.shape[1])
            adj = g2g_hander.constructGraph(hop=1)
            loc_list = None
            start_date = '2002-01-01'
            time = pd.date_range(start_date, periods=data.shape[0], freq = 'D')
            print(f'The last day {time[-1]}')
        elif args.dataset == 'wth2k_hum':
            data_path = os.path.join('./data/weather2k/weather2k_1866x40896x3.npz')
            fdata = np.load(data_path)
            raw_data = fdata['data'] # flow, occupy, speed
            assert raw_data.shape == (1866, 40896, 3), 'shape error: {}'.format(raw_data.shape)
            raw_data = raw_data.transpose(1, 0, 2)
            data = raw_data[:, :, -2]
            assert data.shape == (40896, 1866,), 'shape error: {}'.format(data.shape)
            loc_list = None
            start_date = '2017-01-01 00:00:00'
            time = pd.date_range(start_date, periods=data.shape[0], freq = 'H')
            print(f'The last day {time[-1]}')

            
            latM = fdata['latM']
            lonM = fdata['lonM']
            adj = LatLon_to_adj(latM, lonM)
            adj = adj / 1000.0
        elif dataset == 'PEMS4S':
            data_path = os.path.join('./data/PEMS04S/PEMS04.npz')
            data = np.load(data_path)['data'][:, :, 0]
            start_date = '2018-01-01 00:00:00'
            time = pd.date_range(start_date, periods=data.shape[0], freq = '5MIN')
            adj = None
            loc_list = None
        elif dataset == 'PEMS7S':
            data_path = os.path.join('./data/PEMS07S/PEMS07.npz')
            data = np.load(data_path)['data'][:, :, 0]
            start_date = '2017-05-01 00:00:00'
            time = pd.date_range(start_date, periods=data.shape[0], freq = '5MIN')
            adj = None
            loc_list = None
        elif dataset == 'PEMS8S':
            data_path = os.path.join('./data/PEMS08S/PEMS08.npz')
            data = np.load(data_path)['data'][:, :, 0]
            start_date = '2016-07-01 00:00:00'
            time = pd.date_range(start_date, periods=data.shape[0], freq = '5MIN')
            adj = None
            loc_list = None
        else:
            raise ValueError
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
        print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
        return data, time, adj, loc_list
    
    def build_adj(self, adj_dist):
        adj_ori = pair_dist_to_adj(adj_dist, args.num_nodes, type_='connectivity', self_loop = True, args = args)
        adj_weight = pair_dist_to_adj(adj_dist, args.num_nodes, type_='distance_2', self_loop = True, args = args)
        
        return adj_ori, adj_weight  
          
    
def eval_adj(adj, out_name):
    plt.figure(figsize=(20, 20))
    sns.heatmap(adj, cmap=plt.get_cmap('viridis', 6), center=None, robust=False, square=True, xticklabels=False, yticklabels=False)##30
    plt.tight_layout()
    # plt.savefig('./fig/{}_adj.png'.format(out_name))

def concat_sp_adj(adj_ori):
    pad_adj = np.zeros((int(args.num_nodes), int(args.num_nodes)),                dtype=np.float32)
    adj_row = [adj_ori] +  [pad_adj]*(args.lag -1)
    adj_row = np.concatenate(adj_row, axis=1)
    adj = adj_row
    for idx in range(args.lag - 1):
        adj_new = np.roll(adj_row, args.num_nodes*(idx+1), axis=1)
        adj = np.vstack((adj, adj_new))
    return adj

def concat_temp_adj():
    adj_ori = np.zeros((int(args.num_nodes), int(args.num_nodes)),                dtype=np.float32)
    pad_adj = np.eye(int(args.num_nodes), int(args.num_nodes))
    adj_row = [adj_ori] +  [pad_adj]*(args.lag -1)
    adj_row = np.concatenate(adj_row, axis=1)
    adj = adj_row
    for idx in range(args.lag - 1):
        adj_new = np.roll(adj_row, args.num_nodes*(idx+1), axis=1)
        adj = np.vstack((adj, adj_new))
    return adj
    
