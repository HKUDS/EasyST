import torch
from torch import nn
from torch.autograd import Variable
from .mlp import MultiLayerPerceptron
from numbers import Number
import torch.nn.functional as F

def none_act(inputs):
    return inputs

class DMLP_Stu_IB(nn.Module):

    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.num_nodes = args.num_nodes
        
        self.input_len = args.lag
        self.input_dim = args.input_dim
        if args.dataset in ['PEMS7', 'wth2k_hum']:
            self.output_len = args.horizon
        elif args.dataset == 'CHI_Cri': 
            self.output_len = 4
        self.num_layer = args.num_layer

        self.embed_dim = args.embed_dim
        self.node_dim = args.node_dim
        self.temp_dim_tid = args.temp_dim_tid
        self.temp_dim_diw = args.temp_dim_diw
        self.time_of_day_size = kwargs.get('timeofday')
        self.day_of_week_size = kwargs.get('dayofweek')

        self.if_time_in_day = args.if_T_i_D
        self.if_day_in_week = args.if_D_i_W
        self.if_spatial = args.if_node
        # in out degree
        self.if_in_out = args.if_in_out
        self.in_out_size = args.in_out_size 

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
            if self.if_in_out: 
                self.in_out_feat = nn.Parameter(torch.empty(self.in_out_size, self.node_dim))
                nn.init.xavier_uniform_(self.in_out_feat)
                self.in_out_degree = torch.from_numpy(kwargs.get('in_out_degree'))
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.if_dne = args.if_dne
        hidden_dims = []
        hidden_dims.append(self.embed_dim+self.temp_dim_tid*int(self.if_day_in_week) + self.temp_dim_diw*int(self.if_time_in_day))

        hidden_dims.append(self.node_dim * int(self.if_spatial))
        hidden_dims.append(self.node_dim * int(args.if_dne))
        self.hidden_dim = sum(hidden_dims)
        self.K = int(self.hidden_dim // 2)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.K, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

        ## construct dynamic node encoding: 
        
        if args.if_dne: 
            self.nodevec_p1 = nn.Parameter(torch.randn(self.time_of_day_size, args.mid_dim).to(args.device), requires_grad=True).to(args.device)
            self.nodevec_p2 = nn.Parameter(torch.randn(args.num_nodes, args.mid_dim).to(args.device), requires_grad=True).to(args.device)
            self.nodevec_pk = nn.Parameter(torch.randn(args.mid_dim, args.mid_dim, self.node_dim).to(args.device), requires_grad=True).to(args.device)
            self.dne_emb_layer = nn.Conv2d(
            in_channels= self.input_len, out_channels=1, kernel_size=(1, 1), bias=True)
            
            self.dne_act = {'softplus': F.softplus, 'leakyrelu': nn.LeakyReLU(negative_slope=0.01,inplace=False), 'relu': torch.nn.ReLU(inplace=False), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(dim=2), 'none': none_act}[args.dne_act]

    def construct_dne(self, te): 
        # te: B, T
        assert len(te.shape) == 2, '\'te\' should be (B, T)'
        # print(te.min())
        dne = torch.einsum('bai, ijk->bajk', self.nodevec_p1[te], self.nodevec_pk)
        dne = torch.einsum('bj, cajk->cabk', self.nodevec_p2, dne)
        # B, T, N, D
        dne = self.dne_emb_layer(dne).transpose(3, 1) # B, D, N, 1

        dne = self.dne_act(dne)
        return dne

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        X = history_data[..., range(self.input_dim)] # 
        
        t_i_d_data   = history_data[..., -2] # B, L, N
        d_i_w_data   = history_data[..., -1] # B, L, N

        if self.if_time_in_day:
            T_i_D_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)]    # [B, N, D]
        else:
            T_i_D_emb = None
        if self.if_day_in_week:
            D_i_W_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]          # [B, N, D]
        else:
            D_i_W_emb = None

        # time series embedding
        B, L, N, _ = X.shape # B, L, N, 1
        X = X.transpose(1, 2).contiguous()                      # B, N, L, 1
        X = X.view(B, N, -1).transpose(1, 2).unsqueeze(-1)      # B, L*3, N, 1
        
        time_series_emb = self.time_series_emb_layer(X)         # B, D, N, 1

        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1))  # B, D, N, 1
        if self.if_dne: 
            dne = self.construct_dne(kwargs.get('te').type(torch.LongTensor))
            node_emb.append(dne)
        
        # temporal embeddings
        tem_emb  = []
        if T_i_D_emb is not None:
            tem_emb.append(T_i_D_emb.transpose(1, 2).unsqueeze(-1))                     # B, D, N, 1
        if D_i_W_emb is not None:
            tem_emb.append(D_i_W_emb.transpose(1, 2).unsqueeze(-1))                     # B, D, N, 1
        
        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        hidden = self.encoder(hidden) # B, 256, N, 1
        mu = hidden[:,:self.K, :, :]
        std = F.softplus(hidden[:,self.K:, :, :])
        n_sample = kwargs.get('n_smaple')
        encoding = self.reparametrize_n(mu,std, n = n_sample)
        if n_sample == 1 : pass
        elif n_sample > 1 : encoding = encoding.mean(0)

        hidden_out = encoding
        # regression
        prediction = self.regression_layer(encoding)  
        if self.args.dataset == 'CHI_Cri': 
            prediction = prediction.transpose(3, 1)   

        return prediction, hidden_out, (mu, std)
    def reparametrize_n(self, mu, std, n=1):
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std
    
def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor
