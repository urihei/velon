import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import Quad
# import Simple
import Exp
import Pareto1
import Pareto2


def plot(models, df_list, label, plot_points=False):
    for ind, model in enumerate(models):
        df = df_list[ind]
        plt.figure(ind)
        if plot_points:
            plt.plot(df['x'].values, df['y'].values, '.')
        a = np.sort(df['x'].values)
        plt.plot(a, model.get_y(a), '-', label=label)
        plt.legend()
    plt.show()



def get_data(base_dir, train_test_ind=None, train_percents=0.8):
    files = glob.glob(os.path.join(base_dir, 'data', '*.csv'))
    df_list = []
    df_names = []
    size = 0
    for file_name in files:
        df_list.append(pd.read_csv(file_name))
        df_names.append(file_name)
        size += df_list[-1].shape[0]

    if train_test_ind is None:
        train_test_ind = []
        for df in df_list:
            prm = np.random.permutation(df.shape[0])
            train_test_ind.append((prm[:int(df.shape[0] * train_percents)],
                                   prm[int(df.shape[0] * train_percents):]))
    x_lists = []
    for indx, df in zip(train_test_ind, df_list):
        x_lists.append(df['x'].values[indx[0]])
    x_all = np.concatenate(x_lists, axis=0)
    return train_test_ind, df_list, size, x_all, df_names


def train(cl, base_dir, train_test_ind=None, train_percents=0.95):
    np.random.seed(95)
    train_test_ind, df_list, size, x_all, df_names = get_data(base_dir, train_test_ind, train_percents)
    models = []
    lik = 0
    for indx, df in zip(train_test_ind, df_list):
        models.append(cl(0, 0, 0, 0))
        models[-1].update_a_b(df['x'].values[indx[0]], df['y'].values[indx[0]])
        cur_like = models[-1].get_likelihood(df['x'].values[indx[1]], df['y'].values[indx[1]])
        lik += df.shape[0] / float(size) * cur_like
        print("model:{}, like: {}".format(models[-1].to_string(), cur_like))
    print("Total likelihood:{}".format(lik))

    for _ in xrange(1):
        r_lists = []
        for indx, df, model in zip(train_test_ind, df_list, models):
            r_lists.append(df['y'].values[indx[0]] - model.get_y(df['x'].values[indx[0]]))
        r = np.concatenate(r_lists, axis=0)
        c, d = cl.get_c_d(x_all, r)

        lik = 0
        for indx, df, model in zip(train_test_ind, df_list, models):
            model.set_c_d(c, d)
            model.update_a_b(df['x'].values[indx[0]], df['y'].values[indx[0]])
            cur_like = model.get_likelihood(df['x'].values[indx[1]], df['y'].values[indx[1]])
            lik += df.shape[0] / float(size) * cur_like
            print("model:{}, like:{}".format(model.to_string(), cur_like))
        print("Total likelihood:{}".format(lik))

    return models, train_test_ind, lik, df_names


def test_p2(p1_models, base_dir, train_test_ind=None, train_percents=0.95):
    # Try in user Pareto1 as init for Pareto2
    train_test_ind, df_list, size, x_all = get_data(base_dir=base_dir, train_test_ind=train_test_ind,
                                                    train_percents=train_percents)
    models = []
    for model in p1_models:
        models.append(Pareto2.Pareto2(model.a, model.b, model.c, model.d))
        f = np.log(model.alpha)
        models[-1].set_c_d((model.c, model.d), (0, f))
    r_lists = []

    for indx, df, model in zip(train_test_ind, df_list, models):
        model.update_a_b(df['x'].values[indx[0]], df['y'].values[indx[0]])
        r_lists.append(df['y'].values[indx[0]] - model.get_y(df['x'].values[indx[0]]))

    r = np.concatenate(r_lists, axis=0)
    c, d = Pareto2.Pareto2.get_c_d(x_all, r)
    lik = 0
    for indx, df, model in zip(train_test_ind, df_list, models):
        model.set_c_d(c, d)
        model.update_a_b(df['x'].values[indx[0]], df['y'].values[indx[0]])
        cur_like = model.get_likelihood(df['x'].values[indx[1]], df['y'].values[indx[1]])
        lik += df.shape[0] / float(size) * cur_like
        print("model:{}, like:{}".format(model.to_string(), cur_like))
    print("Total likelihood:{}".format(lik))
    return  models, train_test_ind, lik


def to_csv(models, df_names, file_path=None):
    ab_dic = {}
    for model, path in zip(models, df_names):
        ab_dic[os.path.basename(path)] = model.get_a_b()
    df = pd.DataFrame(ab_dic).T
    df.reset_index(inplace=True)
    df.columns = ['file_name', 'a', 'b']
    if file_path is not None:
        df.to_csv(file_path, index=False)
    return df


def main(base_dir='/home/urihei/proj/'):
    """

    :param base_dir: the base directory - there is data directory that include all the csv files.
    :return: dictionary of the different models and a dictionary with theirs likelihood
    """
    cls = {
#        "simple": Simple.Simple,
        "Normal": Quad.Quad,
        "Exponential": Exp.Exp,
        "Pareto fix alpha": Pareto1.Pareto1,
        "Pareto changed alpha": Pareto2.Pareto2
    }
    models = {}
    likelihoods = {}
    ind = None
    for lab, cl in cls.iteritems():
        print("#########################")
        print("Working on:"+lab)
        models[lab], ind, likelihoods[lab], df_names = train(cl, train_test_ind=ind, base_dir=base_dir)

    return models, likelihoods, df_names
