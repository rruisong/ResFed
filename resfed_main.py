#!/usr/bin/env python
import os
import json
import numpy as np
import torch
import random
import argparse

from datetime import datetime
from datetime import date
from tqdm import tqdm
from client_base import FedClient
from server_base import FedServer
from preprocessing.dataloader import divide_data

from postprocessing.recorder import *
from res_fun.resfed_encoder import *
from res_fun.resfed_compressor import *


def fed_args():
    """
    Arguments for running ResFed
    :return: Arguments for ResFed
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--num_client', type=int, default=10, help='Number of the clients')
    parser.add_argument('-m', '--num_comm_round', type=int, default=300, help='Number of communication rounds')
    parser.add_argument('-c', '--num_local_class', type=int, default=10, help='Number of the classes in each client')
    parser.add_argument('-lr', '--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-bs', '--bs', type=int, default=64, help='Batch size')
    parser.add_argument('-mn', '--model_name', type=str, default='LeNet', help='Model size')
    parser.add_argument('-ds', '--dataset', type=str, default='MNIST',  help='Dataset name')
    parser.add_argument('-rd', '--res_dir', type=str, default='results', help='Directory for the result files')
    parser.add_argument('-e', '--num_epoch', type=int, default=1, help='Number of local epochs')
    parser.add_argument('-is', '--i_seed', type=int, default=1, help='Seed number')
    parser.add_argument('-uls', '--ul_sparsity', type=float, default=0.9, help='Sparsity for deep compression in up-streaming')
    parser.add_argument('-dls', '--dl_sparsity', type=float, default=0.9, help='Sparsity for deep compression in down-streaming')
    parser.add_argument('-ul', '--resfed_upstream', type=int, default=1, help='Flag for ResFed in up-streaming')
    parser.add_argument('-dl', '--resfed_downstream', type=int, default=1,  help='Flag for ResFed in down-streaming')
    parser.add_argument('-qe', '--quan_coef', type=int, default=1, help='Quantization coefficient')
    parser.add_argument('-ulp', '--ul_proto_id', type=int, default=1, help='Prediction rule of the predictor for up-streaming ')
    parser.add_argument('-dlp', '--dl_proto_id', type=int, default=1, help='Prediction rule of the predictor for down-streaming')
    parser.add_argument('-cw', '--comp_weight', type=int, default=0, help='Flag for model weight compression')

    args = parser.parse_args()
    return args

def fed_run():
    """
    Main function for ResFed
    """
    args = fed_args()
    fed_method = 'ResFed'
    np.random.seed(args.i_seed)
    torch.manual_seed(args.i_seed)
    random.seed(args.i_seed)

    today = date.today()
    now = datetime.now().time()
    current_time = today.strftime("%Y%m%d_") + now.strftime("%H%M%S")

    client_dict = {}

    trainset_config, testset = divide_data(num_client=args.num_client, num_local_class=args.num_local_class,
                                           dataset_name=args.dataset, i_seed=args.i_seed)
    recorder = Recorder(trainset_config['users'])
    recorder.res['hyperparam'] = {'lr': args.lr,
                                  'bs': args.bs,
                                  'i_seed': args.i_seed,
                                  'num_epoch': args.num_epoch,
                                  'num_comm_round': args.num_comm_round,
                                  'datetime': current_time,
                                  'fed_method': fed_method,
                                  'dataset_id': args.dataset,
                                  'num_client': args.num_client,
                                  'num_local_class': args.num_local_class,
                                  'Param_01': args.comp_weight,
                                  'Param_02': None,
                                  'Param_03': None,
                                  'Param_04': None,
                                  'model_name': args.model_name,
                                  'ul_spar': args.ul_sparsity,
                                  'dl_spar': args.dl_sparsity,
                                  'resfed_upstream': args.resfed_upstream,
                                  'resfed_downstream': args.resfed_downstream,
                                  'qe': args.quan_coef,
                                  'ul_proto_id': args.ul_proto_id,
                                  'dl_proto_id': args.dl_proto_id
                                  }

    res_filename = current_time + '_' + \
                   args.dataset + '_' + \
                   fed_method + '_' + \
                   str(args.num_local_class) + '_' + \
                   str(args.num_epoch) + '_' + \
                   str(args.i_seed)

    max_acc = 0
    # Initialize the clients for federated learning
    for client_id in trainset_config['users']:

        client_dict[client_id] = FedClient(client_id,
                                           dataset_id=args.dataset,
                                           model_name=args.model_name,
                                           resfed_upstream=args.resfed_upstream,
                                           resfed_downstream=args.resfed_downstream,
                                           ul_proto_id=args.ul_proto_id, dl_proto_id=args.dl_proto_id,
                                           lr=args.lr, num_epoch=args.num_epoch, bs=args.bs)

        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])

    fed_server = FedServer(trainset_config['users'],
                           dataset_id=args.dataset,
                           model_name=args.model_name,
                           resfed_upstream=args.resfed_upstream,
                           resfed_downstream=args.resfed_downstream,
                           ul_proto_id=args.ul_proto_id,
                           dl_proto_id=args.dl_proto_id)
    fed_server.load_testset(testset)
    global_state_dict = fed_server.state_dict()
    global_res_state_dict = {}
    for client_id in trainset_config['users']:
        global_res_state_dict[client_id] = None

    pbar = tqdm(range(args.num_comm_round))
    for global_round in pbar:
        for i, client_id in enumerate(trainset_config['users']):
            if global_round > 0 and args.resfed_downstream:
                recorder.res['clients'][client_id]['dl_bits'].append(calc_msg_size(global_res_state_dict[client_id]))

            else:
                recorder.res['clients'][client_id]['dl_bits'].append(1)
            client_dict[client_id].update(global_state_dict, global_res_state_dict[client_id])

            # Train with the updated model
            state_dict, n_data, loss, res_state_dict = client_dict[client_id].train(global_round, args.ul_sparsity, args.quan_coef)

            # Calc the communication volume for each communication round
            if global_round > 0 and args.resfed_upstream:
                recorder.res['clients'][client_id]['ul_bits'].append(calc_msg_size(res_state_dict))
            else:
                recorder.res['clients'][client_id]['ul_bits'].append(1)

            # Deep weight compression as a baseline for up-streaming
            if args.comp_weight:
                state_dict = sparsify(state_dict, args.ul_sparsity)
                state_dict = quantize(state_dict, args.qe)

            fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, res_state_dict)
        fed_server.select_clients()
        global_state_dict, avg_loss, _, global_res_state_dict = fed_server.agg(args.dl_sparsity, args.quan_coef)

        accuracy = fed_server.test()
        fed_server.flush()

        recorder.res['server']['iid_accuracy'].append(accuracy)
        recorder.res['server']['train_loss'].append(avg_loss)

        if max_acc < accuracy:
            max_acc = accuracy
        pbar.set_description(
            'Global Round: %d' % global_round +
            '| Train loss: %.4f ' % avg_loss +
            '| Accuracy: %.4f' % accuracy +
            '| Max_acc: %.4f' % max_acc)

        if not os.path.exists(args.res_dir):
            os.makedirs(args.res_dir)

        if global_round % 5 == 0:
            with open(os.path.join(args.res_dir, res_filename), "w") as jsfile:
                json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)

    with open(os.path.join(args.res_dir, res_filename), "w") as jsfile:
        json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)


if __name__ == "__main__":
    fed_run()
