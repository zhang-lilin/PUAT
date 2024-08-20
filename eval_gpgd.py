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

DATA_DIR = os.path.join(arg.data_dir, arg.data)
LOG_DIR = os.path.join(arg.log_dir, arg.desc)

attack_name = "gpgd"
cat = 'UAE'
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
        if attack_name in stats[cat] and os.path.exists(test_save_path):
            print('Already tested.')
            exit(0)

if os.path.exists(test_log_path):
    os.remove(test_log_path)
if os.path.exists(test_save_path):
    os.remove(test_save_path)
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
    model_train_seed = cfg_special["seed"]
    logger.log(f"Model training seed {model_train_seed}.")
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
BATCH_SIZE_VALIDATION = arg.batch_size_validation


# To speed up training
torch.backends.cudnn.benchmark = True

# Load data
seed(arg.seed)
test_dataset, test_dataloader = load_test_data(DATA_DIR, batch_size_test=BATCH_SIZE_VALIDATION, num_workers=4)


from core.attacks.ac_gan import Discriminator, Generator
discriminator = Discriminator(info['num_classes']).to(device)
generator = Generator(info['num_classes']).to(device)
checkpoint = torch.load(arg.resume_path)
generator.load_state_dict(checkpoint['netG'])
discriminator.load_state_dict(checkpoint['netD'])
logger.log(f'Resuming conditional GAN at {arg.resume_path}')

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


from core.attacks import create_uae_attack
eval_attack = create_uae_attack(model, generator, discriminator, criterion = nn.CrossEntropyLoss(),
                           attack_type=arg.attack, attack_eps=arg.attack_eps, attack_iter=arg.attack_iter, attack_step=arg.attack_step,
                           rand_init_type='uniform', clip_min=0., clip_max=1.)

seed(arg.seed)
acc, acc_origrn = 0.0, 0.0
model.eval()
origen, adversarial = True, True
total = 0
for _, y in tqdm(test_dataloader, desc='Evaluation : ', disable=False):
    y = y.to(device)
    noise = torch.randn(y.size(0), 100, device=device)
    if origen:
        with torch.no_grad():
            x = generator(noise, y)
            x = transforms.Resize(32)(x)
            out = model(x)
            _, predicted = torch.max(out, 1)
        acc_origrn += (predicted == y).sum().item()
    if adversarial:
        with ctx_noparamgrad_and_eval(model):
            x_adv, _ = eval_attack.perturb(noise, y)
            x_adv = transforms.Resize(32)(x_adv)

        with torch.no_grad():
            out = model(x_adv)
            _, predicted = torch.max(out, 1)
        acc += (predicted == y).sum().item()
    total += x.size(0)

acc /= total
acc_origrn /= total

stats_path = os.path.join(LOG_DIR, 'eval_stats.pkl')
if os.path.exists(stats_path):
    with open(stats_path, "rb") as f:
        logger.stats = pickle.load(f)
if adversarial:
    logger.add(category=cat, k=attack_name, v=acc*100, global_it=model_train_seed, unique=True)
    logger.log('Adversarial {}: {:.5f}%'.format(attack_name, acc*100))
if origen:
    logger.add(category="nat", k="gpgd", v=acc_origrn*100, global_it=model_train_seed, unique=True)
    logger.log('Original: {:.5f}%'.format(acc_origrn*100))

logger.save_stats('eval_stats.pkl')

# CIFAR10_MEAN = torch.tensor(list((0.4914, 0.4822, 0.4465))).reshape(1, 3, 1, 1).cuda()
# CIFAR10_STD = torch.tensor(list((0.2471, 0.2435, 0.2616))).reshape(1, 3, 1, 1).cuda()
# SVHN_MEAN = torch.tensor(list((0.5, 0.5, 0.5))).reshape(1, 3, 1, 1).cuda()
# SVHN_STD = torch.tensor(list((0.5, 0.5, 0.5))).reshape(1, 3, 1, 1).cuda()
# acc, acc_origrn = 0.0, 0.0
# total = 100
# model.eval()
# origen, adversarial = True, True
# seed(arg.seed)
# s = torch.Size([10])
# y = torch.cat([torch.full(s, k) for k in range(10)], 0).to(device)
# # y = torch.randint(0, 10, (BATCH_SIZE_VALIDATION,), device=device, dtype=torch.long)
# noise = torch.randn(total, 100, device=device)
# # noise = torch.cat([noise_ for _ in range(10)], 0).to(device)
# if origen:
#     with torch.no_grad():
#         x = generator(noise, y)
#         x = transforms.Resize(32)(x)
#         out = model(x)
#         _, predicted = torch.max(out, 1)
#         logger.save_images(x, name="gpgd_ori", class_name="img", nrow=10)
#     acc_origrn += (predicted == y).sum().item()
# if adversarial:
#     with ctx_noparamgrad_and_eval(model):
#         x_adv, _ = eval_attack.perturb(noise, y)
#         x_adv = transforms.Resize(32)(x_adv)
#         logger.save_images(x_adv, name="gpgd_uae", class_name="img", nrow=10)
#         pert = x_adv - x
#         std, mean = torch.std_mean(pert)
#         print(std.item(), mean.item())
#         pert = (pert - mean) / std
#         pert = pert * 0.5 + 0.5
#         logger.save_images(pert, name="gpgd_pert", class_name="img", nrow=10)
#     with torch.no_grad():
#         out = model(x_adv)
#         _, predicted = torch.max(out, 1)
#     acc += (predicted == y).sum().item()
# acc /= total
# acc_origrn /= total
# if adversarial:
#     logger.add(category="UAE", k=attack_name, v=acc, global_it=model_train_seed, unique=True)
#     logger.log('Adversarial {}: {:.5f}%'.format(attack_name, acc*100))
# if origen:
#     logger.add(category="nat", k="gpgd", v=acc_origrn, global_it=model_train_seed, unique=True)
#     logger.log('Original: {:.5f}%'.format(acc_origrn*100))

logger.log('\nTesting completed.')