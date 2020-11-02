#!/usr/bin/python
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score

try:
    # For python2
    from itertools import izip as zip
    LARGE_NUMBER = sys.maxsize
except ImportError:
    # For python3
    LARGE_NUMBER = sys.maxsize


# gradient boosting survival tree.
# Input:
# X: input features
# y: Observed time of death(DISCRETE).
# y.shape = [len_input, J]. y[i]=[-1, -1,..., -1, (observed death) 1, 1,
# ..., 1]


class GBSTDataset(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y


class TreeNode(object):
    def __init__(self):
        self.id = None
        self.is_leaf = False
        self.left_child = None
        self.right_child = None
        self.split_feature_id = None
        self.split_val = None
        self.weights = None

    def _calc_split_gain(self, W, V, W_l, V_l, W_r, V_r, lambd):
        """
        Loss reduction
        (Refer to Eq7 of Reference[1])
        """
        def calc_term(w, v):
            return np.square(w) / (v + lambd)
        calced = calc_term(W_l, V_l) + calc_term(W_r, V_r) - calc_term(W, V)
        return np.sum(calced)

    def _calc_leaf_weights(self, ys, r, sigma, lambd):
        """
        Calculate the optimal weights of this leaf node.
        (Refer to Eq5 of Reference[1])
        """
        # need to exclude the "dead" instances before tw_j
        # print("Calc leaf weights on", ys.shape[0], "instances:")
        wl_j_upper = np.zeros([r.shape[1]])
        wl_j_lower = np.zeros([r.shape[1]])

        for index in range(r.shape[0]):
            wl_j_upper += r[index]
            wl_j_lower += sigma[index]

        return -wl_j_upper / (wl_j_lower + lambd)

    def build(self, instances, ys, r, sigma, shrinkage_rate, depth, param):
        """
        Exact Greedy Alogirithm for Split Finidng
        (Refer to Algorithm1 of Reference[1])
        """
        assert instances.shape[0] == ys.shape[0] == len(r) == len(sigma)
        assert ys.shape[1] == r.shape[1] + 1
        if depth >= param['max_depth']:
            self.is_leaf = True
            self.weights = self._calc_leaf_weights(
                ys, r, sigma, param['lambda']) * shrinkage_rate
            return

        W = np.zeros([r.shape[1]], dtype=np.float)
        V = np.zeros([r.shape[1]], dtype=np.float)

        for index in range(r.shape[0]):
            W += r[index]
            V += sigma[index]
        V_sum = np.sum(V)
        best_gain = -LARGE_NUMBER
        best_feature_id = None
        best_val = 0.
        best_left_instance_ids = None
        best_right_instance_ids = None
        for feature_id in range(instances.shape[1]):
            # compute W_l[J] and W_r[J] for all time windows, then sum them up.
            W_l = np.zeros_like(W, dtype=np.float)
            V_l = np.zeros_like(V, dtype=np.float)
            last_V_sum = 0
            V_sum_comp = 0
            sorted_instance_ids = instances[:, feature_id].argsort()
            for i_id in range(sorted_instance_ids.shape[0]):
                V_step = np.zeros_like(V_l)
                W_l += r[sorted_instance_ids[i_id]]
                V_step += sigma[sorted_instance_ids[i_id]]
                V_l += V_step
                last_V_sum += np.sum(V_step)
                if last_V_sum > param['eps'] * V_sum:
                    if i_id != sorted_instance_ids.shape[0] - 1 and \
                            instances[sorted_instance_ids[i_id], feature_id] \
                            == instances[sorted_instance_ids[i_id + 1], feature_id]:
                        # print("encountering same diverging point.")
                        V_sum_comp += np.sum(V_step)
                        continue

                    assert last_V_sum >= 0
                    W_r = W - W_l
                    V_r = V - V_l
                    current_gain = self._calc_split_gain(
                        W, V, W_l, V_l, W_r, V_r, param['lambda'])
                    if current_gain > best_gain:
                        best_gain = current_gain
                        best_feature_id = feature_id
                        best_val = instances[sorted_instance_ids[i_id]
                                             ][feature_id]
                        best_left_instance_ids = sorted_instance_ids.copy()[
                            :i_id + 1]
                        best_right_instance_ids = sorted_instance_ids.copy()[
                            i_id + 1:]
                    last_V_sum = V_sum_comp
                    V_sum_comp = 0
        if best_gain < param['min_split_gain']:
            self.is_leaf = True
            self.weights = self._calc_leaf_weights(
                ys, r, sigma, param['lambda']) * shrinkage_rate
        else:
            self.split_feature_id = best_feature_id
            self.split_val = best_val
            # print("left:", best_left_instance_ids)
            # print("right:", best_right_instance_ids)
            self.left_child = TreeNode()
            self.left_child.build(instances[best_left_instance_ids],
                                  ys[best_left_instance_ids],
                                  r[best_left_instance_ids],
                                  sigma[best_left_instance_ids],
                                  shrinkage_rate,
                                  depth + 1, param)

            self.right_child = TreeNode()

            self.right_child.build(instances[best_right_instance_ids],
                                   ys[best_right_instance_ids],
                                   r[best_right_instance_ids],
                                   sigma[best_right_instance_ids],
                                   shrinkage_rate,
                                   depth + 1, param)

    def predict(self, x):
        if self.is_leaf:
            # print("Ended.")
            return self.weights
            # self.weights is a ndarray with shape [time_steps-1].
        else:
            # print("Splitting on %d, boundary is %f" % (self.split_feature_id, self.split_val))
            if x[self.split_feature_id] <= self.split_val:
                return self.left_child.predict(x)
            else:
                return self.right_child.predict(x)


class Tree(object):
    # Classification and regression tree for tree ensemble

    def __init__(self):
        self.root = None

    def build(self, instances, ys, r, sigma, shrinkage_rate, param):
        assert len(instances) == len(r) == len(sigma)
        self.root = TreeNode()
        current_depth = 0
        self.root.build(
            instances,
            ys,
            r,
            sigma,
            shrinkage_rate,
            current_depth,
            param)

    def predict(self, x):
        return self.root.predict(x)


class GBST(object):
    def __init__(self):
        self.params = {'gamma': 0.,
                       'lambda': 1,
                       'min_split_gain': 0.03,
                       'max_depth': 5,
                       'learning_rate': 1,
                       'eps': 0.2
                       }
        self.best_iteration = None
        self.models = []

    def _calc_training_data_scores(self, train_set):
        """

        :type train_set: GBSTDataset(object)
        """
        if len(self.models) == 0:
            return None
        X = train_set.X
        y = train_set.y
        total_tws = y.shape[1] - 1
        scores = np.zeros([len(X), total_tws])
        for i in range(len(X)):
            scores[i] = self.predict(X[i])
        return scores

    def _calc_gradient(self, train_set, scores):

        hazard_func = np.zeros(
            [train_set.y.shape[0], train_set.y.shape[1] - 1], dtype=np.float)
        weights_hazard_func = np.zeros(
            [train_set.y.shape[1] - 1], dtype=np.float)
        if scores is None:
            # Initializing f(0)
            for tw in range(train_set.y.shape[1] - 1):
                still_alive = len(np.argwhere(train_set.y[:, tw] == -1))
                next_alive = len(np.argwhere(train_set.y[:, tw + 1] == -1))
                hazard_func[:, tw] = (still_alive - next_alive) / (still_alive)
                weights_hazard_func[tw] = (
                    still_alive - next_alive) / (still_alive)
            self.models.append(Tree())
            self.models[0].root = TreeNode()
            self.models[0].root.is_leaf = True
            self.models[0].root.weights = np.log(
                weights_hazard_func / (1 - weights_hazard_func))
        else:
            hazard_func = 1. / (1. + np.exp(-scores))
        # print("hazard:", hazard_func)
        labels = train_set.y
        sigma = (1. - hazard_func) * hazard_func
        r = []
        for i in range(len(hazard_func)):
            r.append(hazard_func[i] - 0.5*(labels[i, 1:] + 1))
            valid_tws = np.count_nonzero(labels[i, :] - 1)
            r[-1][valid_tws+1:] = 0
            sigma[i][valid_tws+1:] = 0
        r = np.array(r)
        # print("Gradient r:", r, "Hessian(sigma):", sigma, sep="\n")
        return r, sigma

    def _build_learner(self, train_set, r, sigma, shrinkage_rate):
        learner = Tree()
        learner.build(
            train_set.X,
            train_set.y,
            r,
            sigma,
            shrinkage_rate,
            self.params)
        return learner

    def calc_auc(self, dataset):
        hazards = []
        for feature in dataset.X:
            hazards.append(self.get_hazard(feature))
        hazards = np.array(hazards)
        mults = np.ones(hazards.shape[0])
        auc_total = []
        for timestep in range(hazards.shape[1]):
            mults = mults * (1 - hazards[:, timestep])
            label = (dataset.y[:, timestep + 1] == -1)
            label = label.astype(np.int)
            try:
                auc = roc_auc_score(y_true=label, y_score=mults)
                # print("AUC Score:", auc)
                auc_total.append(auc)
            except BaseException:
                print("AUC Score: No def.")
        return auc_total

    def train(
            self,
            params,
            train_set,
            test_set,
            num_boost_round=20,
            early_stopping_rounds=5):
        """
        :type test_set: GBSTDataset(object)
        """
        self.params.update(params)
        self.models = []
        shrinkage_rate = 1.
        best_iteration = None
        best_AUC = 0
        train_start_time = time.time()
        print("Train until test scores don't improve for {} rounds.".format(
            early_stopping_rounds))
        for iter_cnt in range(num_boost_round):
            print("--------------------------------")
            iter_start_time = time.time()
            scores = self._calc_training_data_scores(train_set)
            r, sigma = self._calc_gradient(train_set, scores)
            learner = self._build_learner(train_set, r, sigma, shrinkage_rate)
            if iter_cnt > 0:
                shrinkage_rate *= self.params['learning_rate']
            self.models.append(learner)
            print("saving models iter: {}".format(iter_cnt))
            '''
            total_weight = save_models(self.models, params=self.params)
            train_loss = total_weight
            loss = total_weight
            for feature, y in zip(train_set.X, train_set.y):
                f = self.predict(feature)
                for j in range(len(train_set.y[1])-1):
                    if y[j] == -1:
                        train_loss += np.log1p(np.exp(-y[j+1] * f[j]))

            for feature, y in zip(test_set.X, test_set.y):
                f = self.predict(feature)
                for j in range(len(test_set.y[1])-1):
                    if y[j] == -1:
                        loss += np.log1p(np.exp(-y[j+1] * f[j]))
            print("Train Loss: {:.10f}, Test Loss: {:.10f}".format(train_loss, loss))
            '''
            test_auc = np.average(self.calc_auc(test_set))
            print(
                "Average auc on train:",
                np.average(
                    self.calc_auc(train_set)))
            print("Average auc on test:", test_auc)
            print("Iter {:>3}, Elapsed: {:.2f} secs".format(
                iter_cnt, time.time() - iter_start_time))
            if test_auc > best_AUC:
                best_AUC = test_auc
                best_iteration = iter_cnt
            if iter_cnt - best_iteration >= early_stopping_rounds:
                print("Early stopping, best iteration is:")
                print(
                    "Iter {:>3}, AUC: {:.10f}".format(
                        best_iteration, best_AUC))
                break
        self.best_iteration = best_iteration
        save_models(self.models[:best_iteration + 2])
        print("Best iteration is:")
        print("Iter {:>3}, AUC: {:.10f}".format(best_iteration, best_AUC))
        print(
            "Training finished. Elapsed: {:.2f} secs".format(
                time.time() -
                train_start_time))

    def predict(self, x, num_iteration=None):
        """
        generates f[time_windows] for a single input x.
        note: f is not the actual hazard function.
        """
        assert self.models is not None
        if num_iteration is None:
            num_iteration = len(self.models) - 2
        return np.sum((m.predict(x)
                       for m in self.models[:num_iteration + 2]), axis=0)

    def get_hazard(self, x, num_iteration=None):
        predicted = self.predict(x, num_iteration)
        return 1 / (1 + np.exp(-predicted))


def save_models(models, params=None):
    save_lists = []  # models, leaner lists
    total_weights = 0
    for learner in models:
        pre_order = []
        stack = [learner.root]
        node_id = 0
        while stack:
            p = stack.pop()
            p.id = node_id
            node_id += 1
            pre_order.append(p)
            if p.right_child:
                stack.append(p.right_child)
            if p.left_child:
                stack.append(p.left_child)
            if params and p.is_leaf:
                total_weights += 0.5 * \
                    params["lambda"] * np.sum(np.square(p.weights))
        save_list = []  # learner list，[Id, left_child_Id, right_chile_Id, split_feature_id, split_val, weights]
        for i, node in enumerate(pre_order):
            save_list.append([i,
                              node.left_child.id if node.left_child else None,
                              node.right_child.id if node.right_child else None,
                              node.is_leaf,
                              node.split_feature_id,
                              node.split_val,
                              node.weights])
        save_lists.append(save_list)

    np.save('save_models.npy', save_lists)
    print("Save Completed.")
    # print("total weight:", total_weights)
    return total_weights


def load_model():
    save_lists = np.load('save_models.npy', allow_pickle=True)
    models = []
    for save_list in save_lists:
        learner = Tree()
        learner.root = TreeNode()
        learner.root.id = 0
        q = [learner.root]
        while q:
            tmp = []
            for node in q:
                if save_list[node.id][1]:
                    node.left_child = TreeNode()
                    node.left_child.id = save_list[node.id][1]
                    tmp.append(node.left_child)
                if save_list[node.id][2]:
                    node.right_child = TreeNode()
                    node.right_child.id = save_list[node.id][2]
                    tmp.append(node.right_child)
                node.is_leaf, node.split_feature_id, node.split_val, node.weights = save_list[
                    node.id][3:]
            q = tmp
        models.append(learner)
    print("Load Completed.")
    return models
