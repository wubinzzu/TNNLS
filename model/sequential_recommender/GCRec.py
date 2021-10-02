
import numpy as np
import scipy.sparse as sp
from model.AbstractRecommender import SeqAbstractRecommender
from util import DataIterator, timer
from util.tool import csr_to_user_dict_bytime
import tensorflow as tf
from util.cython.random_choice import batch_randint_choice

epsilon = 1e-9


class GCRec(SeqAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(GCRec, self).__init__(dataset, conf)
        self.dataset = dataset
        self.users_num, self.items_num = dataset.train_matrix.shape

        self.lr = conf["lr"]
        self.l2_reg = conf["l2_reg"]
        self.l2_regW = conf["l2_regW"]
        self.factors_num = conf["factors_num"]
        self.seq_L = conf["seq_L"]
        self.seq_T = conf["seq_T"]
        self.neg_samples = conf["neg_samples"]
        self.batch_size = conf["batch_size"]
        self.epochs = conf["epochs"]

        # Capsule's hyperparameters
        self.filter_size = conf["filter_size"]
        self.num_filters = conf["num_filters"]
        self.iter_routing = conf["iter_routing"]
        self.num_outputs_secondCaps = conf["num_outputs_secondCaps"]
        self.vec_len_secondCaps = conf["vec_len_secondCaps"]

        # GCN's hyperparameters
        self.n_layers = conf['n_layers']
        self.norm_adj = self.create_adj_mat(conf['adj_type'])

        self.sess = sess

    @timer
    def create_adj_mat(self, adj_type):
        user_list, item_list = self.dataset.get_train_interactions()
        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.users_num + self.items_num
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.users_num)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            adj_matrix = normalized_adj_single(adj_mat)
            print('use the gcmc adjacency matrix')
        elif adj_type == 'pre':
            # pre adjcency matrix
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
            print('use the pre adjcency matrix')
        else:
            mean_adj = normalized_adj_single(adj_mat)
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        return adj_matrix

    def _create_gcn_embed(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        ego_embeddings = tf.concat([self.embeddings["user_embeddings"], self.embeddings["item_embeddings"]], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            side_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings, name="sparse_dense")

            # transformed sum messages of neighbors.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.users_num, self.items_num], 0)
        return u_g_embeddings, i_g_embeddings

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_placeholder(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.item_seq_ph = tf.placeholder(tf.int32, [None, self.seq_L], name="item_seq")
        self.item_pos_ph = tf.placeholder(tf.int32, [None, self.seq_T], name="item_pos")
        self.item_neg_ph = tf.placeholder(tf.int32, [None, self.neg_samples], name="item_neg")

    def _create_variable(self):
        self.weights = dict()
        self.embeddings = dict()

        embeding_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        user_embeddings = tf.Variable(embeding_initializer([self.users_num, self.factors_num]), dtype=tf.float32)
        self.embeddings.setdefault("user_embeddings", user_embeddings)

        seq_item_embeddings = tf.Variable(embeding_initializer([self.items_num+1, self.factors_num]), dtype=tf.float32)
        self.embeddings.setdefault("seq_item_embeddings", seq_item_embeddings)

        # position embedding
        position_embeddings = tf.Variable(embeding_initializer([self.seq_L, self.factors_num]), dtype=tf.float32)
        self.embeddings.setdefault("position_embeddings", position_embeddings)

        # predication embedding
        item_embeddings = tf.Variable(embeding_initializer([self.items_num, self.factors_num]), dtype=tf.float32)
        self.embeddings.setdefault("item_embeddings", item_embeddings)

        item_biases = tf.Variable(embeding_initializer([self.items_num]), dtype=tf.float32)
        self.embeddings.setdefault("item_biases", item_biases)

        # GCN embedding
        self.user_embeddings, self.item_embeddings = self._create_gcn_embed()

        # Caps embedding
        cap_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
        self.weights.setdefault("caps_conv_W", tf.Variable(cap_initializer([self.seq_L+1, self.filter_size, 1, self.num_filters]), dtype=tf.float32))
        self.weights.setdefault("caps_conv_b", tf.Variable(tf.constant(0.0, shape=[self.num_filters]), dtype=tf.float32))
        
        self.weights.setdefault("routing_W", tf.Variable(cap_initializer([1, self.factors_num,
                                                                          self.num_outputs_secondCaps, self.num_filters,
                                                                          self.vec_len_secondCaps]), dtype=tf.float32))

        # Gate embedding
        Gate_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        self.weights.setdefault("gate_wu", tf.Variable(Gate_initializer([self.factors_num, self.factors_num]), dtype=tf.float32))
        self.weights.setdefault("gate_wi", tf.Variable(Gate_initializer([self.factors_num, self.factors_num]), dtype=tf.float32))
        self.weights.setdefault("gate_wui", tf.Variable(Gate_initializer([self.factors_num, self.factors_num]), dtype=tf.float32))

        self.weights.setdefault("interest_short", tf.Variable(Gate_initializer([self.factors_num, self.factors_num]), dtype=tf.float32))
        self.weights.setdefault("interest_long", tf.Variable(Gate_initializer([self.factors_num, self.factors_num]), dtype=tf.float32))

        self.weights.setdefault("interest_union", tf.Variable(Gate_initializer([self.factors_num, self.factors_num]), dtype=tf.float32))
        self.weights.setdefault("interest_individual", tf.Variable(Gate_initializer([self.factors_num, self.factors_num]), dtype=tf.float32))

        self.weights.setdefault("position_w", tf.Variable(Gate_initializer([self.factors_num, self.factors_num]), dtype=tf.float32))

    def _create_inference(self):
        # embedding lookup
        self.batch_size_b = tf.shape(self.item_seq_ph)[0]

        self.user_embs = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)
        user_embs = tf.expand_dims(self.user_embs, axis=1)

        self.tar_item_emb_pos = tf.nn.embedding_lookup(self.item_embeddings, self.item_pos_ph)
        self.tar_item_emb_neg = tf.nn.embedding_lookup(self.item_embeddings, self.item_neg_ph)

        # position encoding
        position = tf.tile(tf.expand_dims(tf.range(tf.shape(self.item_seq_ph)[1]), 0),
                           [tf.shape(self.item_seq_ph)[0], 1])
        self.position = tf.nn.embedding_lookup(self.embeddings["position_embeddings"], position)

        self.item_embs = tf.nn.embedding_lookup(self.embeddings["seq_item_embeddings"], self.item_seq_ph)
        item_embs = tf.concat([user_embs, self.item_embs], axis=1)
        item_embs = tf.expand_dims(item_embs, axis=-1)

        # The first capsule layer
        with tf.variable_scope('FirstCaps_layer'):
            self.firstCaps = self.CapsLayer(item_embs, layer_type='CONV', kernel_size=1, stride=1)

        # The second capsule layer
        with tf.variable_scope('SecondCaps_layer'):
            self.secondCaps = self.CapsLayer(self.firstCaps, layer_type='FC')
        self.union_level = tf.squeeze(self.secondCaps, axis=-1)

        embedding_Pu = tf.tile(user_embs, tf.stack([1, self.seq_L, 1]))
        embedding_Pu_Qi = embedding_Pu*self.item_embs
        mlp_output = tf.matmul(tf.reshape(embedding_Pu, [-1, self.factors_num]), self.weights["gate_wu"]) + \
                     tf.matmul(tf.reshape(self.item_embs, [-1, self.factors_num]), self.weights["gate_wi"]) + \
                     tf.matmul(tf.reshape(embedding_Pu_Qi, [-1, self.factors_num]), self.weights["gate_wui"])
        mlp_output = tf.reshape(mlp_output, [-1, self.seq_L, self.factors_num])
        mlp_output = tf.nn.sigmoid(mlp_output + tf.matmul(self.position, self.weights["position_w"]))
        gate_short = tf.reduce_sum(tf.multiply(self.item_embs, mlp_output), axis=1, keepdims=True)

        interest_gshort = tf.sigmoid(tf.matmul(self.union_level, self.weights["interest_union"]) +
                                     tf.matmul(gate_short, self.weights["interest_individual"]))
        short = (1-interest_gshort)*self.union_level + interest_gshort*gate_short
        short = tf.squeeze(short, axis=1)

        interest_g = tf.sigmoid(tf.matmul(tf.squeeze(user_embs, axis=1), self.weights["interest_long"], transpose_b=False) +
                                   tf.matmul(short, self.weights["interest_short"], transpose_b=False))
        interest_g = tf.expand_dims(interest_g, axis=1)
        self.output = (1-interest_g)*user_embs + interest_g*tf.expand_dims(short, axis=1)

        tar_item_bias = tf.gather(self.embeddings["item_biases"], tf.concat([self.item_pos_ph, self.item_neg_ph], axis=1))

        return self.output, tar_item_bias

    def _create_loss(self):
        self.p, self.b = self._create_inference()
        tar_item_embs = tf.concat([self.tar_item_emb_pos, self.tar_item_emb_neg], axis=1)
        logits = tf.squeeze(tf.matmul(self.p, tar_item_embs, transpose_b=True), axis=1) + self.b
        pos_logits, neg_logits = tf.split(logits, [self.seq_T, self.neg_samples], axis=1)

        # cross entropy loss
        pos_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos_logits) + 1e-24))
        neg_loss = tf.reduce_sum(-tf.log(1 - tf.sigmoid(neg_logits) + 1e-24))
        loss = pos_loss + neg_loss
        
        self.L2_emb = tf.reduce_sum(tf.square(self.user_embs)) + tf.reduce_sum(tf.square(self.tar_item_emb_pos)) + \
                      tf.reduce_sum(tf.square(self.tar_item_emb_neg)) + tf.reduce_sum(tf.square(self.item_embs)) + \
                      tf.reduce_sum(tf.square(self.b)) + tf.reduce_sum(tf.square(self.position))

        self.L2_weight = tf.reduce_sum(tf.square(self.weights["caps_conv_W"])) + \
                         tf.reduce_sum(tf.square(self.weights["caps_conv_b"])) + \
                         tf.reduce_sum(tf.square(self.weights["routing_W"])) + \
                         tf.reduce_sum(tf.square(self.weights["gate_wu"])) + \
                         tf.reduce_sum(tf.square(self.weights["gate_wi"])) + \
                         tf.reduce_sum(tf.square(self.weights["gate_wui"])) + \
                         tf.reduce_sum(tf.square(self.weights["interest_short"])) + \
                         tf.reduce_sum(tf.square(self.weights["interest_long"])) + \
                         tf.reduce_sum(tf.square(self.weights["interest_union"])) + \
                         tf.reduce_sum(tf.square(self.weights["interest_individual"])) +\
                         tf.reduce_sum(tf.square(self.weights["position_w"]))

        self.Loss_0 = loss + self.l2_reg*self.L2_emb + self.l2_regW*self.L2_weight

        # for predict/test
        self.all_logits = tf.matmul(tf.squeeze(self.p, axis=1), self.item_embeddings, transpose_b=True)

    def _create_optimizer(self):
        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.Loss_0)

    def build_graph(self):
        self._create_placeholder()
        self._create_variable()
        self._create_loss()
        self._create_optimizer()

    def train_model(self):
        self.user_pos_train = csr_to_user_dict_bytime(self.dataset.time_matrix, self.dataset.train_matrix)
        users_list, item_seq_list, item_pos_list = self._generate_sequences()
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            item_neg_list = self._sample_negative(users_list)
            data = DataIterator(users_list, item_seq_list, item_pos_list, item_neg_list,
                                batch_size=self.batch_size, shuffle=True)
            for bat_user, bat_item_seq, bat_item_pos, bat_item_neg in data:
                feed = {self.user_ph: bat_user,
                        self.item_seq_ph: bat_item_seq,
                        self.item_pos_ph: bat_item_pos,
                        self.item_neg_ph: bat_item_neg}

                self.sess.run(self.train_opt, feed_dict=feed)

            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _generate_sequences(self):
        self.user_test_seq = {}
        user_list, item_seq_list, item_pos_list = [], [], []
        userid_set = np.unique(list(self.user_pos_train.keys()))

        for user_id in userid_set:
            seq_items = self.user_pos_train[user_id]
            for index_id in range(len(seq_items)):
                if index_id < self.seq_T: continue
                content_data = list()
                for cindex in range(max([0, index_id - self.seq_L - self.seq_T + 1]), index_id - self.seq_T + 1):
                    content_data.append(seq_items[cindex])

                if (len(content_data) < self.seq_L):
                    content_data = content_data + [self.items_num for _ in range(self.seq_L - len(content_data))]
                user_list.append(user_id)
                item_seq_list.append(content_data)
                item_pos_list.append(seq_items[index_id - self.seq_T + 1:index_id + 1])

            user_id_seq = seq_items[-min([len(seq_items), self.seq_L]):]
            if (len(seq_items) < self.seq_L):
                user_id_seq = user_id_seq + [self.items_num for _ in range(self.seq_L - len(user_id_seq))]
            self.user_test_seq[user_id] = user_id_seq

        return user_list, item_seq_list, item_pos_list

    def _sample_negative(self, users_list):
        neg_items_list = []
        user_neg_items_dict = {}
        all_uni_user, all_counts = np.unique(users_list, return_counts=True)
        user_count = DataIterator(all_uni_user, all_counts, batch_size=1024, shuffle=False)
        for bat_users, bat_counts in user_count:
            n_neg_items = [c*self.neg_samples for c in bat_counts]
            exclusion = [self.user_pos_train[u] for u in bat_users]
            bat_neg = batch_randint_choice(self.items_num, n_neg_items, replace=True, exclusion=exclusion)
            for u, neg in zip(bat_users, bat_neg):
                user_neg_items_dict[u] = neg

        for u, c in zip(all_uni_user, all_counts):
            neg_items = np.reshape(user_neg_items_dict[u], newshape=[c, self.neg_samples])
            neg_items_list.extend(neg_items)
        return neg_items_list

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users, items=None):
        users = DataIterator(users, batch_size=512, shuffle=False, drop_last=False)
        all_ratings = []
        for bat_user in users:
            bat_seq = [self.user_test_seq[u] for u in bat_user]
            feed = {self.user_ph: bat_user,
                    self.item_seq_ph: bat_seq
                    }
            bat_ratings = self.sess.run(self.all_logits, feed_dict=feed)
            all_ratings.extend(bat_ratings)
        all_ratings = np.array(all_ratings, dtype=np.float32)

        if items is not None:
            all_ratings = [all_ratings[idx][item] for idx, item in enumerate(items)]

        return all_ratings

    def CapsLayer(self, input, layer_type, kernel_size=None, stride=None):
        if layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride
            conv = tf.nn.conv2d(
                input,
                self.weights["caps_conv_W"],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            conv1 = tf.nn.relu(tf.nn.bias_add(conv, self.weights["caps_conv_b"]), name="relu")
            conv1 = tf.squeeze(conv1, axis=1)
            capsules = tf.expand_dims(conv1, -1)
            capsules = self.squash(capsules)

            return (capsules)

        if layer_type == 'FC':
            input_R = tf.reshape(input, shape=(-1, input.shape[1].value,
                                               1, input.shape[-2].value, 1))
            with tf.variable_scope('routing'):
                b_ij = tf.constant(np.zeros([self.batch_size, input.shape[1].value, self.num_outputs_secondCaps, 1, 1],dtype=np.float32))
                b_IJ = tf.nn.embedding_lookup(b_ij, tf.shape(input_R)[0])
                capsules = self.routing(input_R, b_IJ, batch_size=self.batch_size_b, iter_routing=self.iter_routing,
                                        num_caps_i=self.factors_num, num_caps_j=self.num_outputs_secondCaps,
                                        len_u_i=self.num_filters, len_v_j=self.vec_len_secondCaps)
                capsules = tf.squeeze(capsules, axis=1)

            return (capsules)

    def routing(self, input_R, b_IJ, batch_size, iter_routing, num_caps_i, num_caps_j, len_u_i, len_v_j):
        input_R = tf.tile(input_R, [1, 1, num_caps_j, 1, 1])
        W = tf.tile(self.weights["routing_W"], [batch_size, 1, 1, 1, 1])

        u_hat = tf.matmul(W, input_R, transpose_a=True)
        u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
        for r_iter in range(iter_routing):
            with tf.variable_scope('iter_' + str(r_iter)):
                c_IJ = tf.nn.softmax(b_IJ, axis=1) * num_caps_i
                if r_iter == iter_routing - 1:
                    s_J = tf.multiply(c_IJ, u_hat)
                    s_J = tf.reduce_sum(s_J, axis=1, keepdims=True)
                    v_J = self.squash(s_J)

                elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                    s_J = tf.multiply(c_IJ, u_hat_stopped)
                    s_J = tf.reduce_sum(s_J, axis=1, keepdims=True)
                    v_J = self.squash(s_J)
                    v_J_tiled = tf.tile(v_J, [1, num_caps_i, 1, 1, 1])
                    u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                    b_IJ += u_produce_v

        return (v_J)

    def squash(self, vector):
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
        vec_squashed = scalar_factor * vector  # element-wise
        return (vec_squashed)