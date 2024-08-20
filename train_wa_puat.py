import json
import time
import shutil
import os
import pandas as pd
import torch
from core.data import get_data_info
from core.data import load_data
from core.data import DATASETS
from core.utils import format_time
from core.utils import Logger
from core.utils import Trainer
from core.utils import seed
from core.utils.watrain_for_puat import WATrainer
from core.utils.config_ import arg, load_config

# Setup
arg.DEFINE_argument('--tau', type=float, default=0.99, help='Weight averaging decay.')
arg.DEFINE_argument('--tau_after', type=float, default=0.999, help='Weight averaging decay.')
arg.DEFINE_argument('-device', '--device', type=str, default='none')
load_config(train=True)
assert arg.data in DATASETS, f'Only data in {DATASETS} is supported!'
arg.unsup_fraction = 0.5
arg.batch_size = 512

DATA_DIR = os.path.join(arg.data_dir, arg.data)
LOG_DIR = os.path.join(arg.log_dir, arg.desc)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
# if os.path.exists(LOG_DIR):
#     shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)
logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if arg.device == 'none':
    arg.device = "cuda" if torch.cuda.is_available() else "cpu"

# device = torch.device("cpu")
# if arg.device == 'none':
#     arg.device = "cpu"

logger.log('Using device: {}'.format(device))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

with open(os.path.join(LOG_DIR, 'arg.txt'), 'w') as f:
    json.dump(arg.get_dict(), f, indent=4)

arg.device = device


info = get_data_info(DATA_DIR)
BATCH_SIZE = arg.batch_size
BATCH_SIZE_VALIDATION = arg.batch_size_validation


# To speed up training
torch.backends.cudnn.benchmark = True
CUDA_LAUNCH_BLOCKING = 1

# Load data
if arg.validation:
    seed(arg.seed)
    train_dataset, test_dataset, eval_dataset, train_dataloader, test_dataloader, eval_dataloader = load_data(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE, batch_size_test=BATCH_SIZE_VALIDATION,
        use_augmentation='none',
        use_consistency=arg.consistency, shuffle_train=True,
        take_amount=arg.take_amount, take_amount_seed=arg.seed,
        aux_data_filename=arg.aux_data_filename,
        num_workers=8,
        unsup_fraction=arg.unsup_fraction,
        validation=True,
        logger=logger
    )
    del train_dataset, test_dataset, eval_dataset
else:
    seed(arg.seed)
    train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE, batch_size_test=BATCH_SIZE_VALIDATION,
        use_augmentation='none',
        use_consistency=arg.consistency, shuffle_train=True,
        take_amount=arg.take_amount, take_amount_seed=arg.seed,
        aux_data_filename=arg.aux_data_filename,
        num_workers=8,
        unsup_fraction=arg.unsup_fraction,
        validation=False,
        logger=logger
    )
    eval_dataloader = None
    del train_dataset, test_dataset


seed(arg.seed)
trainer = WATrainer(info, arg)
last_lr = arg.lr


NUM_ADV_EPOCHS = arg.num_adv_epochs
if arg.debug:
    NUM_ADV_EPOCHS = 1


# Adversarial Training

if arg.resume_path:
    start_epoch = trainer.load_model_puat(arg.resume_path, scheduler=True) + 1
    logger.log(f'Resuming at epoch {start_epoch}')
elif arg.pre_resume_path:
    pre_epoch = trainer.load_model_puat(arg.pre_resume_path, scheduler=False)
    logger.log(f'Resuming at {arg.pre_resume_path} pre-traind for {pre_epoch} epoch')
    trainer.init_optimizer(arg.num_adv_epochs)
    start_epoch = 1
else:
    trainer.init_optimizer(arg.num_adv_epochs)
    start_epoch = 1

if NUM_ADV_EPOCHS > start_epoch:
    logger.log('\n\n')
    metrics = pd.DataFrame()
    if eval_dataloader is not None:
        acc = trainer.eval(eval_dataloader, adversarial=True)
        logger.add('eval', 'adversarial_acc', acc * 100, start_epoch - 1)
        acc = trainer.eval(eval_dataloader, adversarial=False)
        logger.add('eval', 'clean_acc', acc * 100, start_epoch - 1)

    logger.log_info(0, ['test', 'eval'])

    old_score = [0.0, 0.0, 0.0]
    best_epoch = - 1
    logger.log('Adversarial training for {} epochs'.format(NUM_ADV_EPOCHS))
    test_adv_acc = 0.0

for epoch in range(start_epoch, NUM_ADV_EPOCHS+1):
    logger.log('======= Epoch {} ======='.format(epoch))
    if trainer.scheduler:
        last_lr = trainer.scheduler.get_last_lr()[0]
        logger.add('scheduler', 'lr', last_lr, epoch)

    start = time.time()
    res = trainer.train_puat(train_dataloader, epoch=epoch, adversarial=True, logger=logger, verbose=True)
    end = time.time()
    logger.add('time', 'train', format_time(end - start), epoch)

    start_ = time.time()
    test_acc = trainer.eval(test_dataloader)
    # test_adv_acc = trainer.eval(test_dataloader, adversarial=True)
    # logger.add('test', 'adversarial_acc', test_adv_acc * 100, epoch)
    logger.add('test', 'clean_acc', test_acc * 100, epoch)
    # assert 'clean_acc' and 'adversarial_acc' in res
    if 'clean_acc' in res:
        logger.add('train', 'clean_acc', res['clean_acc']*100, epoch)
    if 'adversarial_acc_rae' in res:
        logger.add('train', 'adversarial_acc_rae', res['adversarial_acc_rae']*100, epoch)
    if 'adversarial_acc_uae' in res:
        logger.add('train', 'adversarial_acc_uae', res['adversarial_acc_uae'] * 100, epoch)
    if eval_dataloader:
        eval_adv_acc = trainer.eval(eval_dataloader, adversarial=True)
        eval_acc = trainer.eval(eval_dataloader, adversarial=False)
        logger.add('eval', 'adversarial_acc', eval_adv_acc * 100, epoch)
        logger.add('eval', 'clean_acc', eval_acc * 100, epoch)
        if eval_adv_acc >= old_score[1]:
            old_score[0], old_score[1] = eval_acc, eval_adv_acc
            best_epoch = epoch
            trainer.save_model_puat(WEIGHTS, epoch)
    if (epoch > arg.adv_ramp_start and epoch % arg.adv_eval_freq == 0) or epoch == NUM_ADV_EPOCHS:
        test_adv_acc = trainer.eval(test_dataloader, adversarial=True)
        logger.add('test', 'adversarial_acc', test_adv_acc * 100, epoch)
    if epoch % 1 == 0 or epoch == NUM_ADV_EPOCHS:
        if epoch > arg.adv_ramp_start:
            uae, noise, images = trainer.eval_atk_generator(10)
            logger.save_images(noise, name="noise{:03d}".format(epoch), class_name='img', nrow=10)
            logger.save_images(uae, name="uae{:03d}".format(epoch), class_name='img', nrow=10)
        else:
            images = trainer.eval_generator(10)
        logger.save_images(images, name="img{:03d}".format(epoch), class_name='img', nrow=10)
        logger.plot_loss_gan()
    end_ = time.time()

    if epoch % 10 == 0:
        trainer.save_model_puat(os.path.join(LOG_DIR, 'state-last.pt'), epoch)

    logger.add('time', 'eval', format_time(end_ - start_), epoch)
    cats = ['train', 'test', 'eval', 'scheduler', 'time']
    logger.log_info(epoch, cats)
    logger.plot_learning_curve()
    logger.save_stats('stats.pkl')

logger.log('\nTraining completed.')
if eval_dataloader:
    trainer.load_model_puat(WEIGHTS)
    old_score[2] = trainer.eval(test_dataloader, adversarial=True)
    logger.log('Best checkpoint:  epoch-{}  test-nat-{:.2f}%  eval-adv-{:.2f}%  test-adv-{:.2f}%'.format(best_epoch, old_score[0]*100, old_score[1]*100, old_score[2]*100))
logger.log('Last checkpoint:  epoch-{}  test-nat-{:.2f}%  eval-adv-{:.2f}%  test-adv-{:.2f}%'.format(NUM_ADV_EPOCHS, test_acc*100, eval_adv_acc*100, test_adv_acc*100))