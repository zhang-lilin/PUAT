import json
import time
import pickle
import os
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import argparse
from core.utils.context import ctx_noparamgrad_and_eval
import socket
import torch.nn as nn
from core.models import create_model
from core.data import get_data_info
from core.data import load_test_data
from core.data import DATASETS
from core.utils import format_time
from core.utils import Logger
from core.utils import seed
from core.metrics import accuracy
from core.utils.config_ import arg, load_config

# Setup
arg.DEFINE_argument('-device', '--device', type=str, default='none')
load_config(train=False)
assert arg.data in DATASETS, f'Only data in {DATASETS} is supported!'
arg.attack_eps = 2/255

DATA_DIR = os.path.join(arg.data_dir, arg.data)
LOG_DIR = os.path.join(arg.log_dir, arg.desc)

attack_name, cat = "pgd2", 'RAE'
test_log_path = os.path.join(LOG_DIR, 'log-test-{}.log'.format(attack_name))
test_save_dir = os.path.join(LOG_DIR, 'AEs')
if not os.path.isdir(test_save_dir):
    os.mkdir(test_save_dir)
test_save_path = os.path.join(test_save_dir, f'{attack_name}.pt')
stats_path = os.path.join(LOG_DIR, 'eval_stats.pkl')

if os.path.exists(stats_path):
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    if cat in stats:
        if attack_name in stats['RAE'] and os.path.exists(test_save_path):
            print('Already tested.')
            exit(111)

if os.path.exists(test_log_path):
    os.remove(test_log_path)
logger = Logger(test_log_path)

host_info = "# " + ("%30s" % "Host Name") + ":\t" + socket.gethostname()
logger.log("#" * 120)
logger.log("----------Configurable Parameters In this Model----------")
logger.log(host_info)
for k in arg.get_dict():
    logger.log("# " + ("%30s" % k) + ":\t" + str(arg.__getattr__(k)))
logger.log("#" * 120)

with open(LOG_DIR+'/arg.txt', 'r') as f:
    cfg_special = json.load(f)
    model_train_seed = cfg_special['seed']
    logger.log(f'Model training seed {model_train_seed}.')
    all_keys = arg.get_dict()
    for k in cfg_special:
        if k in all_keys:
            pass
        else:
            v = cfg_special[k]
            if type(v) == bool:
                arg.DEFINE_boolean("-" + k, "--" + k, default=argparse.SUPPRESS)
            else:
                arg.DEFINE_argument(
                    "-" + k, "--" + k, default=argparse.SUPPRESS, type=type(v)
                )
            arg.__setattr__(k, cfg_special[k])
            print("OLD ARG: {} with value {}".format(k, arg.__getattr__(k)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if arg.device == 'none':
    arg.device = "cuda" if torch.cuda.is_available() else "cpu"

logger.log('Using device: {}'.format(device))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

arg.device = device
info = get_data_info(DATA_DIR)
# BATCH_SIZE = arg.batch_size
BATCH_SIZE_VALIDATION = arg.batch_size_validation


# To speed up training
torch.backends.cudnn.benchmark = True

# Load data
seed(arg.seed)
test_dataset, test_dataloader = load_test_data(DATA_DIR, batch_size_test=BATCH_SIZE_VALIDATION, num_workers=4)


if 'standard' not in LOG_DIR:
    WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
else:
    WEIGHTS = os.path.join(LOG_DIR, 'state-last.pt')
model = create_model(arg, info, device)
checkpoint = torch.load(WEIGHTS)
try:
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint['netC'])
    model = torch.nn.DataParallel(model)
except:
    model = torch.nn.DataParallel(model)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint['netC'])
logger.log(f'Resuming target model at {WEIGHTS}')


seed(arg.seed)
acc, acc_origrn = 0.0, 0.0
model.eval()

from core.utils import str2float
from torchattacks import PGD
adversary = PGD(model, eps=2/255, alpha=str2float(arg.attack_step), steps=int(arg.attack_iter))
acc, _, _ = adversary.save(data_loader=test_dataloader, verbose=True, return_verbose=True,
                           save_path=test_save_path)

stats_path = os.path.join(LOG_DIR, 'eval_stats.pkl')
if os.path.exists(stats_path):
    with open(stats_path, "rb") as f:
        logger.stats = pickle.load(f)
logger.add(category=cat, k=attack_name, v=acc, global_it=model_train_seed, unique=True)
logger.log('Adversarial {}: {:.5f}%'.format(attack_name,acc))

logger.save_stats('eval_stats.pkl')
logger.log('\nTesting completed.')