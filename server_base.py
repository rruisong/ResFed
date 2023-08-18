import copy

import torch
import numpy as np
from torch.utils.data import DataLoader
from fed_utilities import assign_dataset, init_model
from res_fun.resfed_predictor import *


class FedServer(object):
    def __init__(self, client_list, dataset_id, model_name, resfed_upstream, resfed_downstream, ul_proto_id, dl_proto_id):
        """
        Initialize the server for federated learning.
        :param client_list: List of the connected clients in networks
        :param dataset_id: Dataset name for the application scenario
        :param model_name: Machine learning model name for the application scenario
        :param resfed_upstream: Flag: 1: Run resfed for upstream, 0: No resfed for upstream
        :param resfed_downstream: Flag: 1: Run resfed for downstream, 0: No resfed for downstream
        :param ul_proto_id: 0: Res-0, 1: Res-1
        :param dl_proto_id: 0: Res-0, 1: Res-1
        """
        self.resfed_downstream = resfed_downstream
        self.resfed_upstream = resfed_upstream
        self.ul_proto_id = ul_proto_id
        self.dl_proto_id = dl_proto_id
        self.glob_trj = {}
        self.locl_trj = {}
        self.client_state = {}
        self.client_loss = {}
        self.client_res_state = {}
        self.client_n_data = {}
        self.selected_clients = []
        self._batch_size = 200
        self.client_list = client_list
        self.testset = None
        self.round = 0
        self.n_data = 0
        self._dataset_id = dataset_id
        gpu = 0
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name,
                                num_class=self._num_class,
                                image_channel=self._image_channel)
        for client_id in client_list:
            self.glob_trj[client_id] = [copy.deepcopy(self.model.state_dict())]
            self.locl_trj[client_id] = []

    def load_testset(self, testset):
        """
        Server loads the test dataset.
        :param data: Dataset for testing.
        """
        self.testset = testset

    def state_dict(self):
        """
        Server returns global model dict.
        :return: Global model dict
        """
        return self.model.state_dict()

    def test(self):
        """
        Server tests the model on test dataset.
        """
        test_loader = DataLoader(self.testset, batch_size=self._batch_size, shuffle=True)
        self.model.to(self._device)
        accuracy_collector = 0
        for step, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                b_x = x.to(self._device)  # Tensor on GPU
                b_y = y.to(self._device)  # Tensor on GPU

                test_output = self.model(b_x)
                pred_y = torch.max(test_output, 1)[1].to(self._device).data.squeeze()
                accuracy_collector = accuracy_collector + sum(pred_y == b_y)
        accuracy = accuracy_collector / len(self.testset)
        self.model.to('cpu')

        return accuracy.cpu().numpy()

    def select_clients(self, connection_ratio=1):
        """
        Server selects a fraction of clients.
        :param connection_ratio: connection ratio in the clients
        """
        # select a fraction of clients
        self.selected_clients = []
        for client_id in self.client_list:
            b = np.random.binomial(np.ones(1).astype(int), connection_ratio)
            if b:
                self.selected_clients.append(client_id)

    def agg(self, dl_spar, qe):
        """
        Server aggregates models from connected clients.
        :param dl_spar: Sparsity for down-streaming.
        :param qe: bit number for parameter.
        :return: model_state: Updated global model after aggregation
        :return: avg_loss: Averaged loss value
        :return: n_data: Number of the local data points
        :return: globel_res_state_dict: Residuals of global models for all clients
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0
        model = init_model(model_name=self.model_name,
                           num_class=self._num_class,
                           image_channel=self._image_channel)
        model_state = model.state_dict()
        avg_loss = 0

        for i, name in enumerate(self.selected_clients):
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:
                if i == 0:
                    model_state[key] = self.client_state[name][key] * self.client_n_data[name] / self.n_data
                else:
                    model_state[key] = model_state[key] + self.client_state[name][key] * self.client_n_data[
                        name] / self.n_data

            avg_loss = avg_loss + self.client_loss[name] * self.client_n_data[name] / self.n_data

        self.model.load_state_dict(model_state)
        self.model.to('cpu')
        globel_res_state_dict = {}

        for client_id in self.client_list:
            if self.resfed_downstream and len(self.glob_trj[client_id]) > 0 and len(self.locl_trj[client_id]) > 1:
                globel_res_state_dict[client_id] = get_residuals(self.locl_trj[client_id],
                                                                 self.glob_trj[client_id],
                                                                 model_state,
                                                                 spar_ratio=dl_spar,
                                                                 qe=qe,
                                                                 proto_id=self.dl_proto_id)
            else:
                globel_res_state_dict[client_id] = None
                self.glob_trj[client_id].append(model_state)
                if len(self.glob_trj[client_id]) > 4:
                    del self.glob_trj[client_id][0]

        self.round = self.round + 1

        return model_state, avg_loss, self.n_data, globel_res_state_dict

    def rec(self, name, state_dict, n_data, loss, res_state_dict):
        """
        Server receives the local updates from the connected client k.
        :param name: Name of client k
        :param state_dict: Model dict from the client k
        :param n_data: Number of local data points in the client k
        :param loss: Loss of local training in the client k
        :param res_state_dict: Model residual dict from the client k
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}

        if self.resfed_upstream and len(self.glob_trj[name]) > 1 and len(self.locl_trj[name]) > 0:
            state_dict = get_recovered(self.glob_trj[name],
                                       self.locl_trj[name],
                                       res_state_dict,
                                       proto_id=self.ul_proto_id)

        else:
            self.locl_trj[name].append(copy.deepcopy(state_dict))
            # avoid memory overheat
            if len(self.locl_trj[name]) > 4:
                del self.locl_trj[name][0]

        self.client_state[name].update(copy.deepcopy(state_dict))
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss

    def flush(self):
        """
        Flushing the client information in the server
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
