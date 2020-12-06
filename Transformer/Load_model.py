import os
import pandas as pd
from main import evaluateIters
from evaluate import ND_Metrics, RMSE_Metrics,Rou_Risk
from transformer_pkg.model import *
from transformer_pkg.utils import *
from transformer_pkg.datasets import *
from transformer_pkg.evaluate import *
from transformer_pkg.optimization import *
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
import torch.nn as nn
import os
from torch import optim
import torch.nn.functional as F

class Arg:
  def __init__(self, dataset,train_ins_num,
               batch_size,
               epoch,
               lr,
               overlap,
               pred_days,
               enc_len,
               dec_len,
               embedded_dim,
               scale_att,
               n_head,
               q_len,
               num_layers,
               print_freq,
               outdir,
               early_stop_ep,
               path,
               weight_decay,
               acc_steps,
               embd_pdrop,
               sparse,
               attn_pdrop,
               resid_pdrop,
               pred_samples,
               optimizer,
               weather_covariates,
               seed=None,
               v_partition=None,

               ):

    self.dataset = dataset
    self.train_ins_num = train_ins_num
    self.batch_size = batch_size
    self.epoch = epoch
    self.lr = lr
    self.overlap = overlap
    self.pred_days = pred_days
    self.enc_len = enc_len
    self.dec_len = dec_len
    self.embedded_dim = embedded_dim
    self.scale_att = scale_att
    self.n_head = n_head
    self.q_len = q_len
    self.num_layers = num_layers
    self.print_freq = print_freq
    self.outdir=outdir
    self.v_partition = v_partition
    self.early_stop_ep = early_stop_ep
    self.seed=seed
    self.path=path
    self.weight_decay=weight_decay
    self.acc_steps=acc_steps
    self.embd_pdrop=embd_pdrop
    self.sparse=sparse
    self.attn_pdrop=attn_pdrop
    self.resid_pdrop=resid_pdrop
    self.pred_samples=pred_samples
    self.optimizer=optimizer
    self.weather_covariates=weather_covariates


def init_encoder():
    v_partition = 0.1 
    dataset = 'solar'
    batch_size = 50
    epoch = 20
    instances = 50000
    lr = 0.001
    overlap = True
    pred_days = 7
    enc_len = 168
    dec_len = 24
    embedded_dim = 20
    weight_decay = 0.0000
    scale_att = True
    n_head = 8
    q_len = 6
    num_layers = 3
    print_freq = 100
    acc_steps = 4
 #Change depending on number of features to be added: input_size = 1+4 date-time+num_features
    dataset = 'solar'
    early_stop_ep = 20
    seed=0
    path = 'solar.csv'
    outdir='results'
    pred_samples=pred_days*24
    optimizer='Adam'
    embd_pdrop = 0.1
    sparse=False
    attn_pdrop = 0.1
    resid_pdrop = 0.1
    weather_covariates='all_visiblity'
    

    args = Arg(dataset=dataset, train_ins_num=instances,
               batch_size=batch_size,
               epoch=epoch,
               lr=lr,
               overlap=overlap,
               pred_days=pred_days,
               enc_len=enc_len,
               dec_len=dec_len,
               embedded_dim=embedded_dim,
               scale_att=scale_att,
               n_head=n_head,
               q_len=q_len,
               num_layers=num_layers,
               print_freq=print_freq,
               outdir=outdir,
               v_partition=v_partition,
               early_stop_ep=early_stop_ep,
               seed=seed,
               path=path,
                weight_decay=weight_decay,
                acc_steps=acc_steps,
               embd_pdrop=0.1,
               sparse=False,
                attn_pdrop = attn_pdrop,
                resid_pdrop = attn_pdrop,
               pred_samples=pred_samples,
               optimizer=optimizer,
            weather_covariates=weather_covariates
               )
    args.input_size=8
    encoder = DecoderTransformer(args,input_dim = args.input_size, n_head= args.n_head, layer= args.num_layers, seq_num = train_set.seq_num , n_embd = args.embedded_dim,win_len= args.enc_len+args.dec_len)
    encoder = nn.DataParallel(encoder).to(device)
    
    return encoder

df_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Pre-trained_Models', 'All_features.pkl'))
outdir='results'
Model = pd.read_pickle(df_path)
Model.load_state_dict(torch.load(outdir+'/19_v_loss.pth.tar')['state_dict'])
#Transformer = init_encoder()
enc_len = 168
dec_len = 24
pred_days = 7
pred_samples = pred_days * 24
df_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Pre-trained_Models', 'test_loader.pkl'))

test_loader = pd.read_pickle(df_path)

predictions = evaluateIters(test_loader, enc_len, dec_len, Model, pred_samples)
ND_metrics = ND_Metrics(predictions, test_loader)
RMSE_metrics = RMSE_Metrics(predictions, test_loader)
rou_metrics = Rou_Risk(0.5, predictions, test_loader, pred_samples)
print('ND: ', ND_metrics)
print('RMSE: ', RMSE_metrics)
print('rou-50: ', rou_metrics)