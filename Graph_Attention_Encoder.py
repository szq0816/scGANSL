import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim
import tensorflow
Dense = tensorflow.keras.layers.Dense
import tensorflow.keras.backend as K
import sys
sys.path.append('..')
from ZINBloss import ZINB


MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def graph_attention_layer(A, M, v, layer):
    with tf.variable_scope("layer_%s" % layer):
        f1 = tf.matmul(M, v[0])
        f1 = A * f1
        f2 = tf.matmul(M, v[1])
        f2 = A * tf.transpose(f2, [1, 0])
        logits = tf.sparse_add(f1, f2)
        unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                  values=tf.nn.sigmoid(logits.values),
                                                  dense_shape=logits.dense_shape)
        attentions = tf.sparse_softmax(unnormalized_attentions)
        attentions = tf.SparseTensor(indices=attentions.indices,
                                     values=attentions.values,
                                     dense_shape=attentions.dense_shape)
        return attentions

def Local_loss(Z, S):
    S = 0.5 * (tf.transpose(S) + S)
    D = tf.linalg.diag(tf.reduce_sum(S, axis=1))
    L = D - S
    Z_transposed = tf.transpose(Z)
    LZ = tf.matmul(L, Z)
    gloss = 2 * tf.linalg.trace(tf.matmul(Z_transposed, LZ))

    return gloss


class GATE:
    def __init__(self, args):
        self.args = args
        hidden_dims, hidden_dims2, lambda_, cell_num, cluster_num = args.hidden_dims_1, args.hidden_dims_2, args.lambda_, args.cell_num, args.cluster_num
        self.lambda_ = lambda_
        self.C = {}
        self.C2 = {}
        self.n_layers = len(hidden_dims) - 1
        self.n_layers2 = len(hidden_dims2) - 1
        self.W, self.v = self.define_weights(hidden_dims)
        self.weight = tf.Variable(1.0e-4 * tf.ones(shape=(cell_num, cell_num)), name="weight")
        self.coef = self.weight - tf.matrix_diag(tf.diag_part(self.weight))  # 减去对角线数值，降低自身相关性影响
        self.params = {"n_clusters": cluster_num, "encoder_dims": [256], "alpha": 1.0}
        self.mu = tf.Variable(tf.zeros(shape=(self.params["n_clusters"], self.params["encoder_dims"][-1])), name="mu")
        self.n_cluster = self.params["n_clusters"]
        self.input_batch_size = cell_num
        self.alpha = self.params['alpha']
        # 新增ZINB模块重构内容
        self.pi = Dense(units=hidden_dims[0], activation='sigmoid', kernel_initializer='glorot_uniform', name='pi')
        self.disp = Dense(units=hidden_dims[0], activation=DispAct, kernel_initializer='glorot_uniform',
                          name='dispersion')
        self.mean = Dense(units=hidden_dims[0], activation=MeanAct, kernel_initializer='glorot_uniform', name='mean')


    def __call__(self, A, A2, X, X2, R, R2, S, S2, p, Theta, Labels):

        # Encoder1
        H = X
        for layer in range(self.n_layers):
            H = self.__encoder(A, H, layer)
        self.H = H
        self.Local_loss1 = Local_loss(self.H, self.coef)
        self.HC = tf.matmul(self.coef, H)
        H = self.HC

        pi1 = self.pi(H)
        disp1 = self.disp(H)
        mean1 = self.mean(H)

        zinb = ZINB(pi1, theta=disp1, ridge_lambda=0, debug=False)
        self.zinb_loss1 = zinb.loss(X, mean1, mean=True)

        # Decoder1
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder(H, layer)
        X_ = H


        layer_flat, num_features = flatten_layer(self.H)
        layer_full = tf.layers.dense(inputs=layer_flat, units=512, activation=None,
                                     kernel_initializer=slim.variance_scaling_initializer(dtype=tf.float32))
        self.z = tf.layers.dense(inputs=layer_full, units=self.n_cluster, activation=None,
                                 kernel_initializer=slim.variance_scaling_initializer(dtype=tf.float32))

        # Encoder2
        H2 = X2
        for layer in range(self.n_layers2):
            H2 = self.__encoder(A2, H2, layer)
        # Final node representations
        self.H2 = H2
        self.Local_loss2 = Local_loss(self.H2, self.coef)

        self.HC2 = tf.matmul(self.coef, H2)
        H2 = self.HC2


        pi2 = self.pi(H2)
        disp2 = self.disp(H2)
        mean2 = self.mean(H2)

        zinb = ZINB(pi2, theta=disp2, ridge_lambda=0, debug=False)
        self.zinb_loss2 = zinb.loss(X2, mean2, mean=True)

        # Decoder2
        for layer in range(self.n_layers2 - 1, -1, -1):
            H2 = self.__decoder(H2, layer)
        X2_ = H2

        layer_flat2, num_features2 = flatten_layer(self.H2)
        layer_full2 = tf.layers.dense(inputs=layer_flat2, units=512, activation=None,
                                      kernel_initializer=slim.variance_scaling_initializer(dtype=tf.float32))
        self.z2 = tf.layers.dense(inputs=layer_full2, units=self.n_cluster, activation=None,
                                  kernel_initializer=slim.variance_scaling_initializer(dtype=tf.float32))

        self.p = p
        self.Theta = Theta
        self.Labels = Labels

        self.Local_loss = self.Local_loss1 + self.Local_loss2

        # The reconstruction loss of node features
        self.features_loss = tf.reduce_sum(tf.pow(tf.subtract(X, X_), 2.0)) + tf.reduce_sum(tf.pow(tf.subtract(X2, X2_), 2.0))

        # The reconstruction loss of the graph structure
        self.S_emb = tf.nn.embedding_lookup(self.H, S)
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        structure_loss1 = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb * self.R_emb, axis=-1)))
        structure_loss1 = tf.reduce_sum(structure_loss1)

        # The reconstruction loss of the graph structure
        self.S_emb2 = tf.nn.embedding_lookup(self.H2, S2)
        self.R_emb2 = tf.nn.embedding_lookup(self.H2, R2)
        structure_loss2 = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb2 * self.R_emb2, axis=-1)))
        structure_loss2 = tf.reduce_sum(structure_loss2)

        self.structure_loss = structure_loss1 + structure_loss2
        self.SE_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.H, self.HC), 2)) + 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.H2, self.HC2), 2))
        self.S_Regular = tf.sqrt(tf.reduce_sum(tf.pow(tf.abs(self.coef), 2.0)))

        # CrossEntropy Loss
        self.cross_entropy1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.z, labels=self.p))
        self.cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.z2, labels=self.p))
        self.dense_loss = self.cross_entropy1 + self.cross_entropy2

        self.Cq_loss = tf.reduce_sum(tf.pow(tf.abs(tf.transpose(self.coef) * self.Theta), 1.0))

        # Pre_train loss
        self.pre_loss = self.features_loss + self.args.lambda_ * self.structure_loss + 10 * self.SE_loss + 1 * self.S_Regular


        self.loss = self.args.lambda_1 * self.features_loss + \
                    self.args.lambda_ * self.structure_loss +\
                    self.args.lambda_2 * self.SE_loss + \
                    self.args.lambda_3 * self.S_Regular + \
                    self.args.lambda_4 * self.Cq_loss + \
                    self.args.lambda_5 * self.dense_loss + \
                    self.args.lambda_6 * self.zinb_loss1 +\
                    self.args.lambda_7 * self.zinb_loss2 + \
                    self.args.lambda_8 * self.Local_loss

        return self.pre_loss, self.loss, self.dense_loss, self.features_loss, self.structure_loss, self.SE_loss, self.coef, self.S_Regular, self.Cq_loss, self.H, self.H2


    def __encoder(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))
        Ws_att = {}
        for i in range(self.n_layers):
            v = {0: tf.get_variable("v%s_0" % i, shape=(hidden_dims[i + 1], 1)),
                 1: tf.get_variable("v%s_1" % i, shape=(hidden_dims[i + 1], 1))}
            Ws_att[i] = v
        return W, Ws_att
