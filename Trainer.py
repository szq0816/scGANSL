import tensorflow as tf

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


from Graph_Attention_Encoder import GATE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
import numpy as np
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
import warnings
from munkres import Munkres
from sklearn import metrics
import datetime


warnings.filterwarnings('ignore')

# Import Metrics NMI, ARI
nmi = normalized_mutual_info_score
ari = adjusted_rand_score
fmi = metrics.cluster.fowlkes_mallows_score


def b3_precision_recall_fscore(labels_true, labels_pred):
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}
    pred_clusters = {}

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score


def f_score(labels_true, labels_pred):
    """Compute the B^3 variant of F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float f_score: calculated F-score
    """
    _, _, f = b3_precision_recall_fscore(labels_true, labels_pred)
    return f

def best_map(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2



def get_one_hot_Label(Label):
    if Label.min() == 0:
        Label = Label
    else:
        Label = Label - 1

    Label = np.array(Label)
    n_class = len(np.unique(Label))
    n_sample = Label.shape[0]
    one_hot_Label = np.zeros((n_sample, n_class))
    for i, j in enumerate(Label):
        one_hot_Label[i, j] = 1

    return one_hot_Label


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


def form_Theta(Q):
    Theta = np.zeros((Q.shape[0], Q.shape[0]))
    for i in range(Q.shape[0]):
        Qq = np.tile(Q[i], [Q.shape[0], 1])
        Theta[i, :] = 1 / 2 * np.sum(np.square(Q - Qq), 1)
    return Theta


def form_structure_matrix(idx, K):
    Q = np.zeros((len(idx), K))
    for i, j in enumerate(idx):
        Q[i, j - 1] = 1
    return Q


def post_proC(C, K, d, alpha):

    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    uu, ss, vv = svds(L, k=K)
    return grp, uu

def thrC(C, ro):

    if ro < 1:

        N = C.shape[1]

        Cp = np.zeros((N, N))

        S = np.abs(np.sort(-np.abs(C), axis=0))

        Ind = np.argsort(-np.abs(C), axis=0)


        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while t < S.shape[0] and not stop:
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C
    return Cp


class Trainer:
    def __init__(self, args):
        self.args = args
        self.build_placeholders()
        self.gate = GATE(args)
        self.pre_cost, self.cost, self.dense_loss, self.features_loss, self.structure_loss,\
        self.SE_loss, self.coef,  self.cRegular, self.cqLoss, self.H, self.H2 \
            = self.gate(self.A, self.A2,
                        self.X, self.X2,
                        self.R, self.R2,
                        self.S, self.S2,
                        self.p, self.Theta, self.Labels)

        self.pre_optimize(self.pre_cost)
        self.optimize(self.cost)
        self.build_session()

    def build_placeholders(self):

        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(tf.int64)
        self.R = tf.placeholder(tf.int64)

        self.A2 = tf.sparse_placeholder(dtype=tf.float32)
        self.X2 = tf.placeholder(dtype=tf.float32)
        self.S2 = tf.placeholder(tf.int64)
        self.R2 = tf.placeholder(tf.int64)

        self.p = tf.placeholder(tf.float32, shape=(None, self.args.cluster_num))
        self.Theta = tf.placeholder(tf.float32, [self.args.cell_num, self.args.cell_num])
        self.Labels = tf.placeholder(tf.int32)

    def build_session(self, gpu=True):
        # GPU Settings
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.allow_growth = False
        # if not gpu:
        #     config.intra_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def pre_optimize(self, pre_loss):
        """
        构建预训练优化器
        """
        pre_optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        pre_gradients, pre_variables = zip(*pre_optimizer.compute_gradients(pre_loss))
        pre_gradients, _ = tf.clip_by_global_norm(pre_gradients, self.args.gradient_clipping)
        self.pre_train_op = pre_optimizer.apply_gradients(zip(pre_gradients, pre_variables))

    def optimize(self, loss):
        """
        构建训练优化器
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables)) #梯度下降法

    def __call__(self, A, A2, X, X2, S, S2, R, R2, L):
        cluster_num = self.args.cluster_num


        Q = form_structure_matrix(L, cluster_num)
        Theta = form_Theta(Q) * 0
        pre_epoch = 0
        while pre_epoch < 1:
            pre_cost, _, pre_st_loss, pre_f_loss, pre_cRegular, pre_SE_loss = self.session.run(
                [self.pre_cost, self.pre_train_op, self.structure_loss, self.features_loss, self.cRegular, self.SE_loss],
                feed_dict={self.A: A, self.A2: A2, self.X: X, self.X2: X2, self.S: S, self.S2: S2, self.R: R,self.R2: R2})

            pre_epoch = pre_epoch + 1
            if pre_epoch % 5 == 0:
                print("-------------------------------------------------------------")
                print("pre_epoch: %d" % pre_epoch, "Pre_Loss: %.2f" % pre_cost, "ReLoss-X: %.2f" % pre_f_loss,"ReLoss-A: %.2f" % pre_st_loss,
                      "pre_SE_loss: %.2f" % pre_SE_loss, "cRegular: %.2f" % pre_cRegular)


        coef = self.session.run(self.coef, feed_dict={self.A: A, self.A2: A2, self.X: X, self.X2: X2, self.S: S, self.S2: S2, self.R: R, self.R2: R2, self.Theta: Theta})
        alpha = max(0.4 - (cluster_num - 1) / 10 * 0.1, 0.1)
        commonZ = thrC(coef, alpha)
        y_x, _ = post_proC(commonZ, cluster_num, 10, 3.5)
        missrate_x = err_rate(L, y_x + 1)
        acc_x = 1 - missrate_x
        print(
            '----------------------------------------------------------------------------------------------------------')
        print("Initial Clustering Results: ")
        print("acc: {:.8f}\t\tnmi: {:.8f}\t\tf_score: {:.8f}\t\tari: {:.8f}".
              format(acc_x, nmi(L, y_x + 1), f_score(L, y_x + 1), ari(L, y_x + 1)))
        print(
            '----------------------------------------------------------------------------------------------------------')
        epoch = 0
        s2_label_subjs = np.array(y_x)
        s2_label_subjs = s2_label_subjs - s2_label_subjs.min() + 1
        s2_label_subjs = np.squeeze(s2_label_subjs)
        one_hot_Label = get_one_hot_Label(s2_label_subjs)

        s2_Q = form_structure_matrix(s2_label_subjs, cluster_num)
        s2_Theta = form_Theta(s2_Q)
        Y = y_x

        # 正式训练循环
        while epoch < self.args.n_epochs:
            cost, _, st_loss, f_loss, CrossELoss, s2_Coef, H, H2 = self.session.run(
                [self.cost, self.train_op, self.structure_loss, self.features_loss, self.dense_loss, self.coef, self.H, self.H2],
                feed_dict={self.A: A, self.A2: A2,
                           self.X: X, self.X2: X2,
                           self.S: S, self.S2: S2,
                           self.R: R,self.R2: R2,
                           self.p: one_hot_Label, self.Theta: s2_Theta, self.Labels: Y})
            cRegular, cqLoss, SELoss = self.session.run(
                [self.cRegular, self.cqLoss, self.SE_loss],
                feed_dict={self.A: A, self.A2: A2,
                           self.X: X, self.X2: X2,
                           self.S: S, self.S2: S2,
                           self.R: R, self.R2: R2,
                           self.p: one_hot_Label, self.Theta: s2_Theta, self.Labels: Y})
            if epoch % (5) == 0:
                s2_label_subjs = np.array(Y)
                s2_label_subjs = s2_label_subjs - s2_label_subjs.min() + 1
                s2_label_subjs = np.squeeze(s2_label_subjs)
                one_hot_Label = get_one_hot_Label(s2_label_subjs)
                s2_Q = form_structure_matrix(s2_label_subjs, cluster_num)
                s2_Theta = form_Theta(s2_Q)

            s2_Coef = thrC(s2_Coef, alpha)

            y_x, Soft_Q = post_proC(s2_Coef, cluster_num, 10, 3.5)
            if len(np.unique(y_x)) != cluster_num:
                continue
            Y = best_map(Y + 1, y_x + 1) - 1
            Y = Y.astype(np.int)

            s2_missrate_x = err_rate(L, Y + 1)

            s2_acc_x = 1 - s2_missrate_x
            s2_nmi_x = nmi(L, Y + 1)
            s2_ari_x = ari(L, Y + 1)
            s2_fmi_x = fmi(L, Y + 1)
            print("epoch: %d" % epoch, "Total_Loss: %.2f" % cost, "ReLoss-X: %.2f" % f_loss, "ReLoss-A: %.2f" % st_loss,
                  "SeLoss: %.2f" % SELoss, "CrossELoss: %.2f" % CrossELoss, "cRegular: %.2f" % cRegular, "cqLoss: %.2f" % cqLoss)
            print("Rearrange:", "scGANSL_Acc:%.4f" % s2_acc_x)
            print("Rearrange:", "scGANSL_Nmi:%.4f" % s2_nmi_x)
            print("Rearrange:", "scGANSL_Ari:%.4f" % s2_ari_x)
            print("Rearrange:", "scGANSL_Fmi:%.4f" % s2_fmi_x)
            print(
                  '----------------------------------------------------------------------------------------------------------')

            epoch = epoch + 1
            result_path = './result_final/' + self.args.dataset + '_Results_xr.txt'
            loss_path = './result_final/' + self.args.dataset + '_Loss_xr.txt'

            start = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
            if epoch == 1:
                fh = open(result_path, 'a')
                ss = '=' * 10 + '新参数：' + start + '=' * 10 + '\n' + str(self.args) + '\n'
                fh.write(ss)
                fh.write('\r\n')
                fh.flush()
                fh.close()

            fh = open(result_path, 'a')
            fh.write('Fin_epoch=%d, ACC=%f, NMI=%f, ADJ_RAND_SCORE=%f, fmi=%f' % (epoch, s2_acc_x, s2_nmi_x, s2_ari_x, s2_fmi_x))
            fh.write('\r\n')
            fh.flush()
            fh.close()

            fh2 = open(loss_path, 'a')
            fh2.write(
                'Fin_epoch=%d, Fin_Total: %.2f, Re_Loss: %.2f, SEloss: %.2f, dense_loss: %.2f, cRegular: %.2f, Cq_loss: %.2f ' % (
                    epoch, cost, f_loss + st_loss, SELoss, CrossELoss,  cRegular, cqLoss))
            fh2.write('\r\n')
            fh2.flush()
            fh2.close()

        return s2_Coef, Y






