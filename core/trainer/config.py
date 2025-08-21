import argparse
import yaml
from .args import _arg_values, _global_parser
from .parser import set_parser_train, set_parser_eval
from runner import ARGS_FOR_TUNE

'''
Argument Definition:
config_file : file to config argument
'''
args = _arg_values()
args.DEFINE_argument("config_file", type=str, help="Path to config file.",)

def load_config(train=True):
    if train:
        set_parser_train(_global_parser)
    else:
        set_parser_eval(_global_parser)

    config_path = args.config_file
    with open(config_path, "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)
    input_arguments = []
    all_keys = args.get_dict()
    print(all_keys)

    for k in cfg_special:
        if k in all_keys and cfg_special[k] == all_keys[k]:
            continue
        else:
            if k in all_keys:
                if k in ARGS_FOR_TUNE:
                    print("Note!: tuning args: {} with value {}".format(k, args.__getattr__(k)))
                    continue
                else:
                    args.__setattr__(k, cfg_special[k])
                    # print("Note!: set args: {} with value {}".format(k, args.__getattr__(k)))
                    input_arguments.append(k)
            else:
                v = cfg_special[k]
                if type(v) == bool:
                    args.DEFINE_boolean("-" + k, "--" + k, default=argparse.SUPPRESS)
                else:
                    args.DEFINE_argument(
                        "-" + k, "--" + k, default=argparse.SUPPRESS, type=type(v)
                    )
                args.__setattr__(k, cfg_special[k])
                print("Note!: new args: {} with value {}".format(k, args.__getattr__(k)))
    if train:
        check(args)
    return input_arguments

import re
def check(args):
    if args.pre_resume_path:
        path = args.pre_resume_path
        path = re.sub('datainfo', f'{args.data}', path)
        path = re.sub('modelinfo', f'{args.model}', path)
        path = re.sub('seedinfo', f'seed{args.seed}', path)
        args.__setattr__('pre_resume_path', path)
