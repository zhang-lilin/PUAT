import argparse

import yaml
from .args import _arg_values, _global_parser
from .parser import set_parser_train, set_parser_eval


'''
Argument Definition:
config_file : file to config argument
'''
arg = _arg_values()
arg.DEFINE_argument("config_file", type=str, help="Path to config file.",)

def load_config(train=True):
    if train:
        set_parser_train(_global_parser)
    else:
        set_parser_eval(_global_parser)

    config_path = arg.config_file
    with open(config_path, "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)
    input_arguments = []
    all_keys = arg.get_dict()
    print(all_keys)

    for k in cfg_special:
        if k in all_keys and cfg_special[k] == all_keys[k]:
            continue
        else:
            if k in all_keys:
                arg.__setattr__(k, cfg_special[k])
                print("Note!: set args: {} with value {}".format(k, arg.__getattr__(k)))
                input_arguments.append(k)
            else:
                v = cfg_special[k]
                if type(v) == bool:
                    arg.DEFINE_boolean("-" + k, "--" + k, default=argparse.SUPPRESS)
                else:
                    arg.DEFINE_argument(
                        "-" + k, "--" + k, default=argparse.SUPPRESS, type=type(v)
                    )
                arg.__setattr__(k, cfg_special[k])
                print("Note!: new args: {} with value {}".format(k, arg.__getattr__(k)))
    if train:
        check(arg)
    return input_arguments


SEEDS = [1, 2, 3]
import re
def check(arg):
    seed_info = f'seed{arg.seed}'
    if 'seed' in arg.desc:
        for s in SEEDS:
            if s != arg.seed:
                s_info = f'seed{s}'
                arg.desc = re.sub(s_info, seed_info, arg.desc)
                arg.pre_resume_path = re.sub(s_info, seed_info, arg.pre_resume_path)

