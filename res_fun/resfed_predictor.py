import copy
import torch
from res_fun.resfed_compressor import *

__all__ = ['get_residuals', 'get_recovered']


def get_residuals(ext_trj, int_trj, state_dict, spar_ratio, qe, pi=1, proto_id=0):
    # Calc the residuals for the to-be-sent model
    # The residuals need to be reused for trajectory synchronization of int_trj
    state_dict = copy.deepcopy(state_dict)
    res_state_dict = None
    with torch.no_grad():
        if len(int_trj) > 0:
            est_state_dict = copy.deepcopy(state_dict)
            res_state_dict = copy.deepcopy(state_dict)
            rec_state_dict = copy.deepcopy(state_dict)
            for key in est_state_dict:
                # predict model
                if proto_id == 0:
                    est_state_dict[key] = int_trj[-1][key]
                elif proto_id == 1:
                    est_state_dict[key] = int_trj[-1][key] - (ext_trj[-2][key] - ext_trj[-1][key]) * 0.5
                else:
                    est_state_dict[key] = (ext_trj[-1][key]
                                           + 2 * (int_trj[-1][key] - ext_trj[-2][key])
                                           - 1 * (int_trj[-2][key] - ext_trj[-3][key]))
                # get residuals
                res_state_dict[key] = state_dict[key] - est_state_dict[key]

            # compress models
            if spar_ratio is not None:
                res_state_dict = sparsify(res_state_dict, spar_ratio)
            if qe is not None:
                res_state_dict = quantize(res_state_dict, qe)

            # sync trajectory
            for key in est_state_dict:
                rec_state_dict[key] = res_state_dict[key] + est_state_dict[key]
            int_trj.append(rec_state_dict)
            if len(int_trj)> 4:
                del int_trj[0]

    return res_state_dict


def get_recovered(int_trj, ext_trj, res_state_dict, pi=1, proto_id=0):
    # recover the received model from residuals
    # The model after recovering should be cached in ext_trj
    res_state_dict = copy.deepcopy(res_state_dict)
    rec_state_dict = None
    with torch.no_grad():
        if len(ext_trj) > 0:
            est_state_dict = copy.deepcopy(res_state_dict)
            rec_state_dict = copy.deepcopy(res_state_dict)
            for key in res_state_dict:
                # predict model
                if proto_id == 0:
                    est_state_dict[key] = ext_trj[-1][key] - (int_trj[-2][key] - int_trj[-1][key]) * 0.5
                elif proto_id == 1:
                    est_state_dict[key] = ext_trj[-1][key]
                else:
                    est_state_dict[key] = (int_trj[-1][key]
                                           + 2 * (ext_trj[-1][key] - int_trj[-2][key])
                                           - 1 * (ext_trj[-2][key] - int_trj[-3][key]))
                # recover model
                rec_state_dict[key] = res_state_dict[key] + est_state_dict[key]
        ext_trj.append(rec_state_dict)
        if len(ext_trj) > 4:
            del ext_trj[0]

    return rec_state_dict
