import os
import random
import json
import matplotlib.pyplot as plt

import numpy as np
import torch


def check_directories(args):
    task_path = os.path.join(args.output_dir)
    if not os.path.exists(task_path):
        os.mkdir(task_path)
        print(f"Created {task_path} directory")

    folder = args.task

    save_path = os.path.join(task_path, folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created {save_path} directory")
    args.save_dir = save_path

    cache_path = os.path.join(args.input_dir, "cache")
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        print(f"Created {cache_path} directory")

    if args.debug:
        args.log_interval /= 10

    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def setup_gpus(args):
    n_gpu = 0  # set the default to 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu
    if n_gpu > 0:  # this is not an 'else' statement and cannot be combined
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return args

def plot_stuff(args, plots1, name1, plots2, name2, plot_type='Loss'):
    if not os.path.isdir("./writeup/plots"):
        os.mkdir("./writeup/plots")

    plt.plot(plots1, label=f'{name1} Set')
    plt.plot(plots2, label=f'{name2} Set')
    plt.title(f"{plot_type} Plot")
    plt.xlabel("Epoch")
    plt.ylabel(plot_type)
    plt.legend()
    plt.savefig("./writeup/plots/" + args.task + "_" + str(args.n_epochs) + "_" + plot_type + ".png")
    plt.clf()

def compare_and_save(args, data):
    if not os.path.isdir(f"./results/{args.task}"):
        os.mkdir(f"./results/{args.task}")
    
    save = {"name": f"{args.task}_{args.n_epochs}"}
    for i in range(len(data)):
        name, d = data[i]

        save[f"{name}_acc"] = d[0]
        save[f"{name}_loss"] = d[1]
    
    json_object = json.dumps(save, indent=4)
    with open(f"./results/{args.task}/result.json", "w") as outfile:
        outfile.write(json_object)