import numpy as np
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
import pickle

json_types = (list, dict, str, int, float, bool, type(None))

color_pool = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]

__all__ = ['PythonObjectEncoder', 'as_python_object', 'Recorder']


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


class Recorder(object):
    def __init__(self, client_list=None):
        self.res_list = []
        self.res = {'hyperparam': {'lr': None,
                                   'bs': None,
                                   'i_seed': None,
                                   'num_epoch': None,
                                   'num_comm_round': None,
                                   'datetime': None,
                                   'fed_method': None,
                                   'dataset_id': None,
                                   'num_client': None,
                                   'num_local_class': None,
                                   'Param_01': None,
                                   'Param_02': None,
                                   'Param_03': None,
                                   'Param_04': None,
                                   'model_name': None,
                                   'ul_spar': None,
                                   'dl_spar': None,
                                   'resfed_upstream': None,
                                   'resfed_downstream': None,
                                   'qe': None,
                                   'ul_proto_id': None,
                                   'dl_proto_id': None
                                   },
                    'server': {'iid_accuracy': [], 'train_loss': []}, 'clients': {}}
        if client_list is not None:
            for client_name in client_list:
                self.res['clients'][client_name] = {'iid_accuracy': [], 'train_loss': [],
                                                    'ul_bits': [],  'dl_bits': [],
                                                    'ul_cr': [], 'dl_cr': []}

    def load(self, filename, label):
        with open(filename) as json_file:
            res = json.load(json_file, object_hook=as_python_object)
        self.res_list.append((res, label))

    def plot(self):
        fig, axes = plt.subplots(3)
        for i, (res, label) in enumerate(self.res_list):
            if i > 6:
                i = 0
            line, = axes[0].plot(np.array(res['server']['iid_accuracy']), '-', label=label, alpha=1, linewidth=3, color=color_pool[i])
            axes[1].plot(np.array(res['server']['train_loss']), '-', label=label, alpha=1, linewidth=3, color=line.get_color())

            for client_id in res['clients']:
                axes[2].plot(np.array(res['clients'][client_id]['ul_bits']), '-', label=label, alpha=0.3, linewidth=1, color=line.get_color())
                axes[2].plot(np.array(res['clients'][client_id]['dl_bits'])*(-1), '-', label=label, alpha=0.3, linewidth=1,
                             color=line.get_color())

        for i, ax in enumerate(axes):
            ax.set_xlabel('Communication Round', size=12)
            if i == 0:
                ax.set_ylabel('Testing Accuracy', size=12)
            if i == 1:
                ax.set_ylabel('Training Loss', size=12)
            if i == 2:
                ax.set_ylabel('Compression Ratio (X)', size=12)
            handles, labels = ax.get_legend_handles_labels()
            newLabels, newHandles = [], []
            for handle, label in zip(handles, labels):
                if label not in newLabels:
                    newLabels.append(label)
                    newHandles.append(handle)
            ax.legend(prop={'size': 12})
            ax.legend(newHandles, newLabels, frameon=True, prop={'size': 12}, )
            ax.grid()
