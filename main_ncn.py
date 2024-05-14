import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertForMaskedLM
import numpy as np

import pandas as pd
import string
import torch.nn.functional as F
from loguru import logger
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import random
from torch_geometric.transforms import NormalizeFeatures
from model import *
from sklearn.metrics import f1_score, roc_curve
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from ncn_model import predictor_dict, convdict, GCN
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from ncn_utils import PermIterator
import time
from typing import Iterable
import os


from utils import *
import argparse
from bert_utils import DesignedTokenizer

from sklearn.model_selection import KFold
from torch_geometric.utils import to_undirected

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)



def get_args():
    
    parser = argparse.ArgumentParser(description='CE-Graph')
    parser.add_argument('--gnn_model', type=str, default='GCNNet', help='Model')
    parser.add_argument('--bert_model_path', default="./trained_model_bert_3", type=str, help='Bert Model')
    parser.add_argument('--vocab_file', default="./bert-config/vocab-3.txt", type=str, help='Vocab File')
    parser.add_argument('--k', default=3, type=int, help='k-mer')
    parser.add_argument('--num_negative_sample', type=int, default=10, help='Number of negative samples')
    
    parser.add_argument('--use_valedges_as_input', action='store_true', help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=500, help="number of epochs")
    
    parser.add_argument('--batch_size', type=int, default=1152, help="batch size")
    parser.add_argument('--maskinput', default=1, action="store_true", help="whether to use target link removal")

    parser.add_argument('--mplayers', type=int, default=1, help="number of message passing layers")
    parser.add_argument('--nnlayers', type=int, default=3, help="number of mlp layers")
    parser.add_argument('--embedding_size', type=int, default=256, help="hidden dimension")
    parser.add_argument('--hiddim', type=int, default=256, help="hidden dimension")
    parser.add_argument('--ln', default=1, action="store_true", help="whether to use layernorm in MPNN")
    parser.add_argument('--lnnn', default=1, action="store_true", help="whether to use layernorm in mlp")
    parser.add_argument('--res', action="store_true", help="whether to use residual connection")
    parser.add_argument('--jk', action="store_true", help="whether to use JumpingKnowledge connection")
    parser.add_argument('--gnndp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--xdp', type=float, default=0.7, help="dropout ratio of gnn")
    parser.add_argument('--tdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--gnnedp', type=float, default=0.0, help="edge dropout ratio of gnn")
    parser.add_argument('--predp', type=float, default=0.3, help="dropout ratio of predictor")
    parser.add_argument('--preedp', type=float, default=0.05, help="edge dropout ratio of predictor")
    parser.add_argument('--gnnlr', type=float, default=0.0043, help="learning rate of gnn")
    parser.add_argument('--prelr', type=float, default=0.0023, help="learning rate of predictor")
    # detailed hyperparameters
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--increasealpha", action="store_true")
    
    parser.add_argument('--splitsize', type=int, default=-1, help="split some operations inner the model. Only speed and GPU memory consumption are affected.")

    # parameters used to calibrate the edge existence probability in NCNC
    parser.add_argument('--probscale', type=float, default=4.3)
    parser.add_argument('--proboffset', type=float, default=2.8)
    parser.add_argument('--pt', type=float, default=0.75)
    parser.add_argument("--learnpt", action="store_true")

    # For scalability, NCNC samples neighbors to complete common neighbor. 
    parser.add_argument('--trndeg', type=int, default=-1, help="maximum number of sampled neighbors during the training process. -1 means no sample")
    parser.add_argument('--tstdeg', type=int, default=-1, help="maximum number of sampled neighbors during the test process")
    # NCN can sample common neighbors for scalability. Generally not used. 
    parser.add_argument('--cndeg', type=int, default=-1)
    
    # predictor used, such as NCN, NCNC
    parser.add_argument('--predictor', default="cn0", choices=predictor_dict.keys())
    parser.add_argument("--depth", type=int, default=1, help="number of completion steps in NCNC")
    # gnn used, such as gin, gcn.
    parser.add_argument('--model', choices=convdict.keys())

    parser.add_argument('--save_gemb', action="store_true", help="whether to save node representations produced by GNN")
    parser.add_argument('--load', type=str, help="where to load node representations produced by GNN")
    parser.add_argument("--loadmod", action="store_true", help="whether to load trained models")
    parser.add_argument("--savemod", action="store_true", help="whether to save trained models")
    
    parser.add_argument("--savex", action="store_true", help="whether to save trained node embeddings")
    parser.add_argument("--loadx", action="store_true", help="whether to load trained node embeddings")
    
    parser.add_argument("--testbs", default=8192, help="batch size for testing")
    
    return parser.parse_args()

def get_rna_sequence_embedding(model, tokenizer, sequence, layer_index=-2, device='cuda:0'):
    """
    This function takes an RNA sequence and returns the embedding from the specified BERT layer.
    
    Args:
        model (PreTrainedModel): The trained BERT model.
        tokenizer (PreTrainedTokenizer): The tokenizer used with the BERT model.
        sequence (str): The RNA sequence to embed.
        layer_index (int): The index of the layer from which to extract the embedding. 
                           By default, it uses the second to last layer.

    Returns:
        numpy.ndarray: The embedding of the RNA sequence.
    """
    # Tokenize the sequence and convert to input IDs
    inputs = tokenizer(sequence, return_tensors="pt", max_length=512, truncation=True, padding='max_length')

    # Extract input IDs and attention mask
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get the model's output (make sure the model is in evaluation mode)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    # Extract the hidden states from the specified layer
    hidden_states = outputs.hidden_states[layer_index]

    # For simplicity, use mean pooling across the token embeddings to get a single sequence embedding
    # Adjust this pooling method based on your specific needs or preferences
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_hidden = torch.sum(hidden_states * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)  # Avoid division by zero
    mean_hidden = sum_hidden / sum_mask

    # Convert the tensor to a NumPy array and return it
    return mean_hidden.cpu().numpy()

def get_rna_sequence_embedding_v2(model, tokenizer, sequence, device='cuda:0'):
    """
    This function takes an RNA sequence and returns the embedding from the specified BERT layer.
    
    Args:
        model (PreTrainedModel): The trained BERT model.
        tokenizer (PreTrainedTokenizer): The tokenizer used with the BERT model.
        sequence (str): The RNA sequence to embed.
        layer_index (int): The index of the layer from which to extract the embedding. 
                           By default, it uses the second to last layer.

    Returns:
        numpy.ndarray: The embedding of the RNA sequence.
    """
    # Tokenize the sequence and convert to input IDs
    inputs = tokenizer(sequence, return_tensors="pt", max_length=512, truncation=True, padding='max_length')

    # Extract input IDs and attention mask
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get the model's output (make sure the model is in evaluation mode)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    # Extract the hidden states from the specified layer
    hidden_states = outputs.hidden_states[-1]

    sentence_embedding = hidden_states[:, 0, :].squeeze()

    # Convert the tensor to a NumPy array and return it
    return sentence_embedding.cpu().numpy()

def load_data(path="./datasets/ceRNA"):
    # Replace these with your actual data loading and processing
    match_table = pd.read_csv(f"{path}/circRNA_lncrna_miRNA_interaction.csv", index_col=0)
    graph_table = pd.read_csv(f"{path}/index_value.csv")
    # node_table = pd.read_csv(f"{path}/node_link.csv")
    node_table = pd.read_csv(f"{path}/node_link_with_label.csv")
    return match_table, graph_table, node_table


def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None,
          num_negative_samples: int=1):
    
    if alpha is not None:
        predictor.setalpha(alpha)
    
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), 
                                data.adj_t.sizes()[0],
                                num_neg_samples=num_negative_samples * pos_train_edge.size(1)).to(pos_train_edge.device)
    for order, perm in enumerate(PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    )):
        # print(order)
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        h = model(data.x, adj)
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss


@torch.no_grad()
def test(model, predictor, data, split_edge, batch_size, interaction_type):
    model.eval()
    predictor.eval()

    # Depending on the interaction_type, select the appropriate edges
    if interaction_type == 'lncRNA-miRNA':
        pos_test_edge = split_edge['test']['lncRNA_miRNA']['edge'].to(data.adj_t.device())
        neg_test_edge = split_edge['test']['lncRNA_miRNA']['edge_neg'].to(data.adj_t.device())
    elif interaction_type == 'circRNA-miRNA':
        pos_test_edge = split_edge['test']['circRNA_miRNA']['edge'].to(data.adj_t.device())
        neg_test_edge = split_edge['test']['circRNA_miRNA']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)

    pos_test_pred = torch.cat([
        predictor(h, adj, pos_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0], batch_size, False)
    ], dim=0)

    neg_test_pred = torch.cat([
        predictor(h, adj, neg_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0], batch_size, False)
    ], dim=0)
     
    pos_test_pred = pos_test_pred.flatten()
    neg_test_pred = neg_test_pred.flatten()
    ypred = torch.cat((pos_test_pred, neg_test_pred), dim=0).cpu().numpy()
    ytrue =  torch.cat((torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)), dim=0).cpu().numpy()
    aucscore = roc_auc_score(ytrue, ypred)

    fpr, tpr, thresholds = roc_curve(ytrue, ypred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_new = (ypred >= optimal_threshold).astype(int)
    f1 = f1_score(ytrue, y_pred_new)
    
    avg_precision = average_precision_score(ytrue, ypred)
    ndcg = NDCG(ytrue, ypred)
    
    return f1, aucscore, avg_precision, ndcg


def removerepeated(ei):
    ei = to_undirected(ei)
    ei = ei[:, ei[0]<ei[1]]
    return ei


if __name__ == "__main__":

    args = get_args()
    
    gnn_model = args.model

    vocab_file = args.vocab_file
    k = args.k
    embedding_size = args.embedding_size

    num_negative_sample = args.num_negative_sample

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    date = "20240504"
    
    if not os.path.exists(f"log/{date}"):
        os.makedirs(f"log/{date}")
    
    
    log_path = f"log/{date}/file_bertrna_ncn_{gnn_model}_{k}_{num_negative_sample}_{args.predictor}_{embedding_size}.log"

    logger.add(log_path, rotation="10 MB")
    
    logger.info(f"log file path is {log_path}")
    
    match_table, graph_table, node_table = load_data()

    unique_lnc_circ = list(match_table['circrna_or_lncrna'].unique())
    unique_mi = list(match_table['mirna'].unique())

    lnc_circ_seq = []
    mi_seq = []

    for i in unique_lnc_circ:
        seq = match_table[match_table['circrna_or_lncrna'] == i]["circrna_or_lncrna_seq"]
        seq = list(seq)
        seq = seq[0]
        seq = seq.translate(str.maketrans('', '', string.punctuation))
        lnc_circ_seq.append(seq)

    for i in unique_mi:
        seq = match_table[match_table['mirna'] == i]["mirna_seq"]
        seq = list(seq)
        seq = seq[0]
        seq = seq.replace('.', '')
        if ',' in seq:
            seq = seq.split(',')
            seq = seq[0]

        mi_seq.append(seq)
        
    all_name = unique_lnc_circ + unique_mi
    all_sequences = lnc_circ_seq + mi_seq
        
    logger.info("[INFO] Finish getting the sequence of lncRNA, circRNA, and miRNA")    

    
    if not os.path.exists(f"checkpoints/bert_{k}_embedding.pth"):
        logger.info(f"Using the bert model from {args.bert_model_path}")
        # Load the trained model from directory
        model = BertForMaskedLM.from_pretrained(args.bert_model_path).to(device)
        tokenizer = DesignedTokenizer(vocab_file=vocab_file)
        vectors = {}
        for name, seq in tqdm(zip(all_name, all_sequences), total=len(all_sequences)):
            vectors[name] = get_rna_sequence_embedding_v2(model, tokenizer, seq)
            
        # Create Graph Embeddings
        graph_label = list(graph_table["rna"])
        graph_embedding = np.zeros((len(graph_label), embedding_size))
        for node, vec in vectors.items():
            position = graph_label.index(node)
            graph_embedding[position] = vec
        x_embedding = torch.tensor(graph_embedding).float() 
        
        logger.info("Saving the x_embedding")
        torch.save(x_embedding, f"checkpoints/bert_{k}_embedding.pth")
    else:
        x_embedding = torch.load(f"checkpoints/bert_{k}_embedding.pth")
        logger.info("x_embedding loaded")
        

     # # Construct DGL Graph
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # best_epoches = []
    # f1_scores = []
    # auc_scores = []
    # ap_scores = []
    # ndcg_scores = []
    
    lncrna_mirna_f1_scores = []
    lncrna_mirna_auc_scores = []
    lncrna_mirna_ap_scores = []
    lncrna_mirna_ndcg_scores = []
    
    circrna_mirna_f1_scores = []
    circrna_mirna_auc_scores = []
    circrna_mirna_ap_scores = []
    circrna_mirna_ndcg_scores = []
        
    for train_index, test_index in kfold.split(node_table):
        # Direct conversion to integer indices and elimination of redundant list conversions
        train_set = node_table.iloc[train_index]
        test_set = node_table.iloc[test_index]
        
        logger.info("train_set columns is {}".format(train_set.columns))

        # Simplify the construction of undirected graph edges for training and validation sets
        u_train, v_train = train_set['circrna_or_lncrna_index'].to_numpy(dtype=int), train_set['mirna_index'].to_numpy(dtype=int)
        u_test, v_test = test_set['circrna_or_lncrna_index'].to_numpy(dtype=int), test_set['mirna_index'].to_numpy(dtype=int)
        
        # Only select circrna_or_lncrna_label == lncrna as u_test_lncrna_mirna, v_test_lncrna_mirna
        u_test_lncrna_mirna, v_test_lncrna_mirna = test_set[test_set['circrna_or_lncrna_label'] == 'lncrna']['circrna_or_lncrna_index'].to_numpy(dtype=int), test_set[test_set['circrna_or_lncrna_label'] == 'lncrna']['mirna_index'].to_numpy(dtype=int)
        
        # Only select circrna_or_lncrna_label == circrna as u_test_circrna_mirna, v_test_circrna_mirna
        u_test_circrna_mirna, v_test_circrna_mirna = test_set[test_set['circrna_or_lncrna_label'] == 'circrna']['circrna_or_lncrna_index'].to_numpy(dtype=int), test_set[test_set['circrna_or_lncrna_label'] == 'circrna']['mirna_index'].to_numpy(dtype=int)
        
        u_train = torch.from_numpy(u_train)
        v_train = torch.from_numpy(v_train)
        u_test = torch.from_numpy(u_test)
        v_test = torch.from_numpy(v_test)
        
        u_test_lncrna_mirna = torch.from_numpy(u_test_lncrna_mirna)
        v_test_lncrna_mirna = torch.from_numpy(v_test_lncrna_mirna)
        
        u_test_circrna_mirna = torch.from_numpy(u_test_circrna_mirna)
        v_test_circrna_mirna = torch.from_numpy(v_test_circrna_mirna)
        

        # For undirected graphs, edges are bidirectional so we concatenate them in both directions
        # Utilize PyTorch's functionality to streamline this process
        edge_index_train = torch.tensor(np.concatenate([u_train, v_train, v_train, u_train]), dtype=torch.long).view(2, -1)
        edge_index_test = torch.tensor(np.concatenate([u_test, v_test, v_test, u_test]), dtype=torch.long).view(2, -1)
        
        edge_index_test_lncrna_mirna = torch.tensor(np.concatenate([u_test_lncrna_mirna, v_test_lncrna_mirna, v_test_lncrna_mirna, u_test_lncrna_mirna]), dtype=torch.long).view(2, -1)
        edge_index_test_circrna_mirna = torch.tensor(np.concatenate([u_test_circrna_mirna, v_test_circrna_mirna, v_test_circrna_mirna, u_test_circrna_mirna]), dtype=torch.long).view(2, -1)
        
        # Test graph
        test_data = Data(x=x_embedding, 
                        edge_index=edge_index_test, 
                        edge_label=torch.ones(len(u_test)),
                        edge_label_index=torch.stack([u_test, v_test], dim=0)).to(device)
        neg_edge_index_test = negative_sampling(edge_index=edge_index_test, 
                                            num_nodes=test_data.num_nodes, 
                                            num_neg_samples=args.num_negative_sample * test_data.edge_label_index.size(1), method='sparse').to(device)
        # Perform negative sampling for validation data
        
        test_data.edge_label_index = torch.cat([test_data.edge_label_index, neg_edge_index_test], dim=-1)
        test_data.edge_label = torch.cat([test_data.edge_label, torch.zeros(neg_edge_index_test.size(1), dtype=torch.float).to(device)], dim=0)
        
        
        test_data_lncrna_mirna = Data(x=x_embedding,
                        edge_index=edge_index_test_lncrna_mirna,
                        edge_label=torch.ones(len(u_test_lncrna_mirna)),
                        edge_label_index=torch.stack([u_test_lncrna_mirna, v_test_lncrna_mirna], dim=0)).to(device)
        neg_edge_index_test_lncrna_mirna = negative_sampling(edge_index=edge_index_test_lncrna_mirna,
                                            num_nodes=test_data_lncrna_mirna.num_nodes,
                                            num_neg_samples=args.num_negative_sample * test_data_lncrna_mirna.edge_label_index.size(1), method='sparse').to(device)
        # Perform negative sampling for validation data
        test_data_lncrna_mirna.edge_label_index = torch.cat([test_data_lncrna_mirna.edge_label_index, neg_edge_index_test_lncrna_mirna], dim=-1)
        test_data_lncrna_mirna.edge_label = torch.cat([test_data_lncrna_mirna.edge_label, torch.zeros(neg_edge_index_test_lncrna_mirna.size(1), dtype=torch.float).to(device)], dim=0)
        
        
        test_data_circrna_mirna = Data(x=x_embedding,
                        edge_index=edge_index_test_circrna_mirna,
                        edge_label=torch.ones(len(u_test_circrna_mirna)),
                        edge_label_index=torch.stack([u_test_circrna_mirna, v_test_circrna_mirna], dim=0)).to(device)
        neg_edge_index_test_circrna_mirna = negative_sampling(edge_index=edge_index_test_circrna_mirna,
                                            num_nodes=test_data_circrna_mirna.num_nodes,
                                            num_neg_samples=args.num_negative_sample * test_data_circrna_mirna.edge_label_index.size(1), method='sparse').to(device)
        # Perform negative sampling for validation data
        test_data_circrna_mirna.edge_label_index = torch.cat([test_data_circrna_mirna.edge_label_index, neg_edge_index_test_circrna_mirna], dim=-1)
        test_data_circrna_mirna.edge_label = torch.cat([test_data_circrna_mirna.edge_label, torch.zeros(neg_edge_index_test_circrna_mirna.size(1), dtype=torch.float).to(device)], dim=0)
        
        
        # Add the logic of the NCN
        split_edge = {'train': {}, 'test': {}}
        split_edge['train']['edge'] = removerepeated(edge_index_train).t()
        split_edge['test']['edge'] = removerepeated(edge_index_test).t()
        split_edge['test']['edge_neg'] = removerepeated(neg_edge_index_test).t()
        
        split_edge['test']['lncRNA_miRNA'] = {}
        split_edge['test']['lncRNA_miRNA']['edge'] = removerepeated(edge_index_test_lncrna_mirna).t()
        split_edge['test']['lncRNA_miRNA']['edge_neg'] = removerepeated(neg_edge_index_test_lncrna_mirna).t()
        
        split_edge['test']['circRNA_miRNA'] = {}
        split_edge['test']['circRNA_miRNA']['edge'] = removerepeated(edge_index_test_circrna_mirna).t()
        split_edge['test']['circRNA_miRNA']['edge_neg'] = removerepeated(neg_edge_index_test_circrna_mirna).t()
        
        
        
        # Prepare the Data objects for training and validation with neg_sampling for the validation set
        train_data = Data(x=x_embedding, edge_index=edge_index_train, edge_label=torch.ones(len(u_train)), edge_label_index=torch.stack([u_train, v_train], dim=0)).to(device)
        train_data.edge_index = to_undirected(split_edge['train']['edge'].t())
        train_edge_index = train_data.edge_index
        train_data.edge_weight = None
        train_data.adj_t = SparseTensor.from_edge_index(train_edge_index, 
                                                        sparse_sizes=(train_data.num_nodes, train_data.num_nodes)).to(device)
        train_data.adj_t = train_data.adj_t.to_symmetric().coalesce()
        train_data.max_x = -1
        train_data.full_adj_t = train_data.adj_t

        # Apply feature normalization to both datasets
        normalize_features = NormalizeFeatures()
        train_data = normalize_features(train_data)
        test_data = normalize_features(test_data)
        
        
        predfn = predictor_dict[args.predictor]
        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                            args.predp, args.preedp, args.lnnn).to(device)
        model = GCN(train_data.num_features, 
                args.hiddim, args.hiddim, args.mplayers,
                        args.gnndp, args.ln, args.res, train_data.max_x,
                        args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
                {'params': predictor.parameters(), 'lr': args.prelr}])
        
        # Train and Test
        results = []
        lncrna_mirna_results = []
        circrna_mirna_results = []
        
        # best_test_auc = 0
        
        for epoch in tqdm(range(1, 501)):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            loss = train(model, predictor, train_data, split_edge, optimizer,
                            args.batch_size, args.maskinput, [], alpha, args.num_negative_sample)
            # test_f1, test_auc, test_ap, test_ndcg = test(model, predictor, train_data, split_edge,
            #                 args.testbs)
            lncrna_mirna_test_f1, lncrna_mirna_test_auc, lncrna_mirna_test_ap, lncrna_mirna_test_ndcg = test(model, predictor, train_data, split_edge, args.testbs, 'lncRNA-miRNA')
            circrna_mirna_test_f1, circrna_mirna_test_auc, circrna_mirna_test_ap, circrna_mirna_test_ndcg = test(model, predictor, train_data, split_edge, args.testbs, 'circRNA-miRNA')
            # if test_auc > best_test_auc:
            #     best_test_auc = test_auc
            # results.append([epoch, test_f1, test_auc, test_ap, test_ndcg])
            lncrna_mirna_results.append([epoch, lncrna_mirna_test_f1, lncrna_mirna_test_auc, lncrna_mirna_test_ap, lncrna_mirna_test_ndcg])
            circrna_mirna_results.append([epoch, circrna_mirna_test_f1, circrna_mirna_test_auc, circrna_mirna_test_ap, circrna_mirna_test_ndcg])
            
        # best_result = max(results, key=lambda x: x[2])
        
        best_result_lncrna_mirna = max(lncrna_mirna_results, key=lambda x: x[2])
        best_result_circrna_mirna = max(circrna_mirna_results, key=lambda x: x[2])
        
        # logger.info('{} Best result: Epoch: {}, F1: {:.3f}, AUC: {:.3f}, AP: {:.3f}, NDCG: {:.3f}'.format(gnn_model,
        #         *best_result))
        
        logger.info('{} Best result lncRNA-miRNA: Epoch: {}, F1: {:.3f}, AUC: {:.3f}, AP: {:.3f}, NDCG: {:.3f}'.format(gnn_model,
                *best_result_lncrna_mirna))
        logger.info('{} Best result circRNA-miRNA: Epoch: {}, F1: {:.3f}, AUC: {:.3f}, AP: {:.3f}, NDCG: {:.3f}'.format(gnn_model,
                                                                                                                        *best_result_circrna_mirna))
        
        # best_epoches.append(best_result[0])
        # f1_scores.append(best_result[1])
        # auc_scores.append(best_result[2])
        # ap_scores.append(best_result[3])
        # ndcg_scores.append(best_result[4])
        
        lncrna_mirna_f1_scores.append(best_result_lncrna_mirna[1])
        lncrna_mirna_auc_scores.append(best_result_lncrna_mirna[2])
        lncrna_mirna_ap_scores.append(best_result_lncrna_mirna[3])
        lncrna_mirna_ndcg_scores.append(best_result_lncrna_mirna[4])
        
        circrna_mirna_f1_scores.append(best_result_circrna_mirna[1])
        circrna_mirna_auc_scores.append(best_result_circrna_mirna[2])
        circrna_mirna_ap_scores.append(best_result_circrna_mirna[3])    
        circrna_mirna_ndcg_scores.append(best_result_circrna_mirna[4])
    
    # I want to save the checkpoint of the model for the further testing
    logger.info("Saving the model")
    torch.save(model.state_dict(), f"checkpoints/{gnn_model}_model_ncn.pth")
    logger.info("Saving the predictor")
    torch.save(predictor.state_dict(), f"checkpoints/{gnn_model}_predictor_ncn.pth")
    

    # logger.info("-----------------------Final Results-----------------------")
    # logger.info('{} F1 scores: mean {:.3f}, std {:.3f}'.format(
    #     gnn_model, 
    #     np.mean(f1_scores), np.std(f1_scores)))
    # logger.info('{} AUC scores: mean {:.3f}, std {:.3f}'.format(
    #     gnn_model,
    #     np.mean(auc_scores), np.std(auc_scores)))
    # logger.info('{} AP scores: mean {:.3f}, std {:.3f}'.format(
    #     gnn_model,
    #     np.mean(ap_scores), np.std(ap_scores)))
    # logger.info('{} NDCG scores: mean {:.3f}, std {:.3f}'.format(
    #     gnn_model,
    #     np.mean(ndcg_scores), np.std(ndcg_scores)))
    # logger.info("-----------------------------------------------------------")

    logger.info("-----------------------Final Results lncRNA-miRNA-----------------------")
    logger.info('{} F1 scores: mean {:.3f}, std {:.3f}'.format(
        gnn_model, 
        np.mean(lncrna_mirna_f1_scores), np.std(lncrna_mirna_f1_scores)))
    logger.info('{} AUC scores: mean {:.3f}, std {:.3f}'.format(gnn_model,
        np.mean(lncrna_mirna_auc_scores), np.std(lncrna_mirna_auc_scores)))
    logger.info('{} AP scores: mean {:.3f}, std {:.3f}'.format(gnn_model,
        np.mean(lncrna_mirna_ap_scores), np.std(lncrna_mirna_ap_scores)))
    logger.info('{} NDCG scores: mean {:.3f}, std {:.3f}'.format(gnn_model,
        np.mean(lncrna_mirna_ndcg_scores), np.std(lncrna_mirna_ndcg_scores)))
    logger.info("-----------------------------------------------------------")
    
    logger.info("-----------------------Final Results circRNA-miRNA-----------------------")
    logger.info('{} F1 scores: mean {:.3f}, std {:.3f}'.format(
        gnn_model, 
        np.mean(circrna_mirna_f1_scores), np.std(circrna_mirna_f1_scores)))
    logger.info('{} AUC scores: mean {:.3f}, std {:.3f}'.format(gnn_model,
        np.mean(circrna_mirna_auc_scores), np.std(circrna_mirna_auc_scores)))
    logger.info('{} AP scores: mean {:.3f}, std {:.3f}'.format(gnn_model,
        np.mean(circrna_mirna_ap_scores), np.std(circrna_mirna_ap_scores)))
    logger.info('{} NDCG scores: mean {:.3f}, std {:.3f}'.format(gnn_model,
        np.mean(circrna_mirna_ndcg_scores), np.std(circrna_mirna_ndcg_scores)))
    logger.info("-----------------------------------------------------------")