import copy
from models.models import *
import numpy as np

from torch.utils.data import DataLoader
from fed_utilities import assign_dataset, init_model

from res_fun.resfed_predictor import *


class FedClient(object):
    def __init__(self, name, dataset_id, model_name, resfed_upstream, resfed_downstream, ul_proto_id, dl_proto_id, lr, num_epoch, bs):
        """
        Initialize the client k for federated learning.
        :param name: Name of the client k
        :param dataset_id: Local dataset in the client k
        :param model_name: Local model in the client k
        :param resfed_upstream: Flag: 1: Run resfed for upstream, 0: No resfed for upstream
        :param resfed_downstream: Flag: 1: Run resfed for downstream, 0: No resfed for downstream
        :param ul_proto_id: 0: Res-0, 1: Res-1
        :param dl_proto_id: 0: Res-0, 1: Res-1
        :param lr: Learning rate
        :param num_epoch: Number of local training epochs in the client k
        :param bs: Batch size used in the client k
        """

        # Initialize the metadata in the local client
        self.target_ip = '127.0.0.3'
        self.port = 9999
        self.name = name
        self._epoch = num_epoch
        self._batch_size = bs
        self._lr = lr
        self._momentum = 0.9
        self.num_workers = 2
        self.loss_rec = []
        self.n_data = 0
        self.global_round = 0

        # Initialize the ResFed in the local client
        self.resfed_downstream = resfed_downstream
        self.resfed_upstream = resfed_upstream
        self.ul_proto_id = ul_proto_id
        self.dl_proto_id = dl_proto_id
        self.glob_trj = []
        self.locl_trj = []

        # Initialize the dataset in the local client
        self.trainset = None
        self.test_data = None

        # Initialize the local model
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.param_len = sum([np.prod(p.size()) for p in model_parameters])
        self.est_model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel) #

        # Training on GPU
        gpu = 0
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    def load_trainset(self, trainset):
        """
        Client loads the training dataset.
        :param trainset: Dataset for training.
        """
        self.trainset = trainset
        self.n_data = len(trainset)

    def update(self, model_state_dict, res_state_dict):
        """
        Client updates the model from the server.
        :param model_state_dict: Global model.
        """
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)

        if self.resfed_downstream and len(self.glob_trj) > 0 and len(self.locl_trj) > 1:
            model_state_dict = get_recovered(self.locl_trj, self.glob_trj, res_state_dict, proto_id=self.dl_proto_id)
        else:
            self.glob_trj.append(copy.deepcopy(model_state_dict))
            if len(self.glob_trj) > 4:
                del self.glob_trj[0]

        self.model.load_state_dict(model_state_dict)

    def train(self, global_round, ul_spar, qe):
        """
        Client trains the model on local dataset
        :param global_round: Global communication round.
        :param ul_spar: Sparsity for up-streaming.
        :param qe: bit number for parameter.
        :return Local updated model.
        :return number of local data points.
        :return training loss.
        :return residuals.
        """
        self.global_round = global_round
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        self.model.to('cpu')

        res_state_dict = None
        if self.resfed_upstream and len(self.glob_trj) > 1 and len(self.locl_trj) > 0:
            res_state_dict = get_residuals(self.glob_trj, self.locl_trj, self.model.state_dict(), spar_ratio=ul_spar, qe=qe, proto_id=self.ul_proto_id)

        else:
            self.locl_trj.append(copy.deepcopy(self.model.state_dict()))
            # Avoid memory overheat
            if len(self.locl_trj) > 4:
                del self.locl_trj[0]

        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy(), res_state_dict


