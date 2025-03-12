import sys
sys.path.append("")

import os.path
import torch
import random
import argparse
import json
import numpy as np

from datetime import datetime

from trainer import Trainer

seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

project_path = os.getcwd()
print(project_path)

def get_params():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--devices', type=bool, default=True)
    parser.add_argument("--algorithm", type=str, default="HyperSED")
    parser.add_argument('--mode', type=str, default="open_set")
    parser.add_argument('--data_path', type=str, default=project_path + '/data_preprocess')
    parser.add_argument('--save_model_path', type=str, default='./saved_models')
    parser.add_argument('--n_cluster_trials', type=int, default=5)
    parser.add_argument('--dataset_name', type=str, default='Event2012', choices=['Event2012', 'Event2018'])
    parser.add_argument('--encode', type=str, default='SBERT', choices=['SBERT', 'BERT'])
    parser.add_argument('--edge_type', type=str, default='e_as', choices=['e_as', 'e_a', 'e_s'])
    parser.add_argument('--gpu', type=int, default=3, help='gpu')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--hgae', type=bool, default=True)
    parser.add_argument('--dsi', type=bool, default=True)
    parser.add_argument('--pre_anchor', type=bool, default=True)
    parser.add_argument('--anchor_rate', type=int, default=20)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--thres', type=float, default=0.5)
    parser.add_argument('--diag', type=float, default=0.5)

    # HGAE
    parser.add_argument('--num_layers_gae', type=int, default=2)
    parser.add_argument('--hidden_dim_gae', type=int, default=128)
    parser.add_argument('--out_dim_gae', type=int, default=2)
    parser.add_argument('--t', type=float, default=1., help='Fermi-Dirac decoder')
    parser.add_argument('--r', type=float, default=2., help='Fermi-Dirac decoder')
    parser.add_argument('--lr_gae', type=float, default=1e-3)
    parser.add_argument('--w_decay', type=float, default=0.3)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--nonlin', type=str, default=None)
    parser.add_argument('--use_attn', type=bool, default=False)
    parser.add_argument('--use_bias', type=bool, default=True)

    # DSI, H-dimensional structural information
    parser.add_argument('--decay_rate', type=float, default=None)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=2)
    parser.add_argument('--height', type=int, default=2)
    parser.add_argument('--max_nums', type=int, nargs='+', default=[300], help="such as [50, 7]")   # pre-defined node cluster num, 因为大部分为null，所以也无所谓了！
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--lr_pre', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()

    return args


def main(args):
    results = []
    times = []
    args.algorithm_name = '_'.join([name for name, value in vars(args).items() 
                                if value and name in ['hgae', 'pre_anchor', 'dsi']])

    if args.mode == 'closed_set':
        trainer = Trainer(args)
        trainer.train_model()
        results = trainer.result
        times = trainer.time

    else:
        if args.dataset_name == 'Event2012':
            blocks = [20]
            # blocks = [i+1 for i in range(21)]
            for block in blocks:
                # try:
                trainer = Trainer(args, block)
                trainer.train_model()
                results.append(trainer.result)
                times.append(trainer.time)
                # except:
                #     continue
        else:
            blocks = [i+1 for i in range(16)]
            for block in blocks:
                try:
                    trainer = Trainer(args, block)
                    trainer.train_model()
                    results.append(trainer.result)
                    times.append(trainer.time)
                except:
                    continue

    # saving args and results
    time = datetime.now().strftime('%m%d%H%M%S')
    save_path = f"./para_results/{args.dataset_name}/{args.mode}"
    time_save_path = f"./running_times/{args.dataset_name}/{args.mode}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(time_save_path):
        os.makedirs(time_save_path)
    results_list = results
    times_list = times
    args_dict = vars(args)
    data = {'results': results_list,
            'args':args_dict}
    data_json = json.dumps(data, indent=4)
    times_json = json.dumps(times_list, indent=4)
    with open(f"{save_path}/{time}_{args.algorithm_name}.json", "w") as json_file:
        json_file.write(data_json)
    with open(f"{time_save_path}/{time}_{args.algorithm_name}.json", "w") as json_file:
        json_file.write(times_json)

    torch.cuda.empty_cache()


if __name__ == '__main__':

    main(get_params())