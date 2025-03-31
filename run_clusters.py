import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tensorflow as tf
from Trainer import Trainer
import process
from utils import load_data, load_graph, buildGraphNN
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import preprocessH5
import numpy as np


def parse_args():
    """
    Parses the arguments.
    """

    parser = argparse.ArgumentParser(description="Run gate.")
    parser.add_argument('--dataset', nargs='?', default='Klein', help='Input dataset')
    parser.add_argument('--cell_num', type=int, help='The number of cell.')
    parser.add_argument('--cluster_num', type=int, help='The number of cell type.')
    parser.add_argument('--gene_num', default=1000, type=int, help='The number of gene.')
    parser.add_argument('--k_NN', default=30, type=int, help='The number of KNN.')
    parser.add_argument('--seed', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate. Default is 0.001.')
    parser.add_argument('--n-epochs', default=30, type=int, help='Number of epochs')
    parser.add_argument('--hidden-dims-1', type=list, nargs='+', default=[512, 512], help='Number of dimensions1.')
    parser.add_argument('--hidden-dims-2', type=list, nargs='+', default=[512, 512], help='Number of dimensions1.')
    parser.add_argument('--lambda-', default=1, type=float, help='Parameter controlling the contribution of edge '
                                                                  'Graph reconstruction in the loss function.')
    parser.add_argument('--lambda-1', default=0, type=float, help= 'Parameter controlling the contribution of features_loss.')
    parser.add_argument('--lambda-2', default=10, type=float, help='Parameter controlling the contribution of SE_loss.')
    parser.add_argument('--lambda-3', default=1, type=float, help='Parameter controlling the contribution of S_Regular.')
    parser.add_argument('--lambda-4', default=5, type=float, help='Parameter controlling the contribution of Cq_loss.')
    parser.add_argument('--lambda-5', default=5, type=float, help='Parameter controlling the contribution of dense_loss.')
    parser.add_argument('--lambda-6', default=0.01, type=float, help='Parameter controlling the contribution of zinb_loss1.')
    parser.add_argument('--lambda-7', default=0.01, type=float, help='Parameter controlling the contribution of zinb_loss2.')
    parser.add_argument('--lambda-8', default=0.5, type=float, help='Parameter controlling the contribution of local_gloss.')
    parser.add_argument('--dropout', default=0.0, type=float, help='Dropout.')
    parser.add_argument('--gradient_clipping', default=3.0, type=float, help='gradient clipping')

    return parser.parse_args()



def main(args):
    """
    Load Dataset.
    """

    if args.dataset in {'Klein'}:

        filename = "dataset/" + args.dataset + "/data.h5"

        X, Label, _ ,X_row, _, _ = preprocessH5.load_h5(filename, args.gene_num)
        pca = PCA(n_components=X.shape[1])
        X2 = pca.fit_transform(X_row)

        b_test = MinMaxScaler()  # 训练数据，赋值给b_test
        X2 = b_test.fit_transform(X2)
        G = buildGraphNN(X, args.k_NN)


    # prepare the data
    G_tf, S, R = process.prepare_graph_data(G)
    G_tf2 = G_tf
    S2 = S
    R2 = R

    # add feature dimension size to the beginning of hidden_dims
    feature_dim1 = X.shape[1]
    args.hidden_dims_1 = [feature_dim1] + args.hidden_dims_1
    feature_dim2 = X2.shape[1]
    args.hidden_dims_2 = [feature_dim2] + args.hidden_dims_2
    args.cell_num = X.shape[0]
    args.cluster_num = len(np.unique(Label))
    print('Dim_hidden_1: ' + str(args.hidden_dims_1))
    print('Dim_hidden_2: ' + str(args.hidden_dims_2))
    trainer = Trainer(args)
    coef,y_pred = trainer(G_tf, G_tf2, X, X2, S, S2, R, R2, Label)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(args.seed)


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    main(args)

