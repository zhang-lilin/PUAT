import json
import os
import shutil
import time
import pandas as pd
import torch
from core.trainer import *
from core.trainer.config import args, load_config
from core.data import get_data_info, load_data
from core.models import create_model

# Setup
load_config(train=True)
args.desc = f'{args.method}_{args.model}_{args.data}_seed{args.seed}'

DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')

resume_path = None
if os.path.exists(LOG_DIR):
    print("File exists already.")
    resume_path = os.path.join(LOG_DIR, 'state-last.pt')
    if os.path.exists(resume_path):
        print('Try loading from the last checkpoint in the exist file. ')
        logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))
        logger.iflog = False
    else:
        print("No checkpoint saved.")
        shutil.rmtree(LOG_DIR)
        resume_path = None
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))
    logger.iflog = True

# device =  "cpu"
# logger.log(f'Using device: cpu')
# args.device = "cpu"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.log(f'Using device: {torch.cuda.get_device_name(device)}')
args.device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
CUDA_LAUNCH_BLOCKING=1

ARGS_FILE = os.path.join(LOG_DIR, 'args.txt')
if os.path.exists(ARGS_FILE) and resume_path is not None:
    with open(ARGS_FILE, 'r') as f:
        cfg_special = json.load(f)
        all_keys = args.get_dict()
        for k in cfg_special:
            v = cfg_special[k]
            if v == all_keys[k]:
                pass
            else:
                logger.log(f"ARG CONFLICT for {k}: old-{v} new-{all_keys[k]}")
else:
    with open(ARGS_FILE, 'w') as f:
        json.dump(args.get_dict(), f, indent=4)

args.device = device
info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation

# To speed up training
torch.backends.cudnn.benchmark = True

logger.log("### " + ("%30s" % "DATASET"))
if args.pre_resume_path:
    pseudo_label_model = create_model(args=args, info=info, device=device, logger=logger)
    path = args.pre_resume_path
    checkpoint = torch.load(path)
    if 'model_state_dict' not in checkpoint:
        raise RuntimeError('Model weights not found at {}.'.format(path))
    try:
        pseudo_label_model.load_state_dict(checkpoint['model_state_dict'])
    except:
        pseudo_label_model = torch.nn.DataParallel(pseudo_label_model)
        pseudo_label_model.load_state_dict(checkpoint['model_state_dict'])
    pre_epoch = checkpoint['epoch']
    logger.log(f'Pseudo labels by {args.pre_resume_path} (pre-traind for {pre_epoch} epoch)')
else:
    pseudo_label_model = None

# Load data
seed(args.seed)
train_dataset, test_dataset, eval_dataset, train_dataloader, test_dataloader, eval_dataloader = load_data(
    data_dir=DATA_DIR, batch_size=BATCH_SIZE, batch_size_test=BATCH_SIZE_VALIDATION,
    use_augmentation=args.augment if 'puat' not in args.method else 'none',
    pseudo_label_model=pseudo_label_model,
    take_amount=args.take_amount, take_amount_seed=args.seed,
    unsup_fraction=args.unsup_fraction, validation=args.validation,
    num_workers=0, logger=logger,
)
del train_dataset, test_dataset, eval_dataset
args.num_batches = len(train_dataloader)


logger.log("### " + ("%30s" % "TRAINER"))
seed(args.seed)
trainer = Trainer(info, args, logger=logger, dataloader=train_dataloader)
if args.pre_resume_path:
    trainer.load_model(args.pre_resume_path, False)

NUM_ADV_EPOCHS = args.num_epochs
# Adversarial Training
if NUM_ADV_EPOCHS > 0:
    logger.log('\n\n')
    old_score = [0.0, 0.0]
    logger.log('Adversarial training for {} epochs'.format(NUM_ADV_EPOCHS))

if resume_path is not None:
    if eval_dataloader and os.path.exists(WEIGHTS):
        best_epoch = trainer.load_model(WEIGHTS, load_opt=True)
        test_acc = trainer.eval(test_dataloader, adversarial=False)
        eval_adv_acc = trainer.eval(eval_dataloader, adversarial=True)
        old_score[0], old_score[1] = test_acc, eval_adv_acc
        logger.log(f'Best checkpoint resuming at epoch {best_epoch}. ')
    start_epoch = trainer.load_model(resume_path, load_opt=True) + 1
    logger.log(f'Resuming at epoch {start_epoch - 1}')
else:
    start_epoch = 1


if NUM_ADV_EPOCHS >= start_epoch:
    logger.iflog = True
    metrics = pd.DataFrame()
    test_acc = trainer.eval(test_dataloader, adversarial=False)
    logger.add('test', 'clean_acc', test_acc * 100, start_epoch - 1)
    eval_acc = 0.0
    if eval_dataloader:
        eval_acc = trainer.eval(eval_dataloader, adversarial=False)
        logger.add('eval', 'clean_acc', eval_acc * 100, start_epoch - 1)
        eval_adv_acc = trainer.eval(eval_dataloader, adversarial=True)
        logger.add('eval', 'adversarial_acc', eval_adv_acc * 100, start_epoch - 1)
        best_epoch = -1
        old_score = [0.0, 0.0, 0.0]
    logger.log_info(0, ['test', 'eval'])

for epoch in range(start_epoch, NUM_ADV_EPOCHS+1):
    logger.log('======= Epoch {} ======='.format(epoch))
    if trainer.scheduler is not None:
        last_lr = trainer.scheduler.get_last_lr()[0]
        logger.add('scheduler', 'lr', last_lr, epoch)
    start = time.time()
    res = trainer.train(train_dataloader, epoch=epoch, verbose=True)
    for k in res:
        if 'acc' in k:
            logger.add('train', k, res[k] * 100, epoch)
    end = time.time()
    logger.add('time', 'train', format_time(end - start), epoch)
    start_ = time.time()
    if eval_dataloader:
        eval_acc = trainer.eval(eval_dataloader, adversarial=False)
        logger.add('eval', 'clean_acc', eval_acc * 100, epoch)
        eval_adv_acc = trainer.eval(eval_dataloader, adversarial=True)
        logger.add('eval', 'adversarial_acc', eval_adv_acc * 100, epoch)
        if eval_adv_acc >= old_score[1]:
            old_score[0], old_score[1] = test_acc, eval_adv_acc
            trainer.save_model(WEIGHTS, epoch)
            best_epoch = epoch
    test_acc = trainer.eval(test_dataloader, adversarial=False, verbose=True)
    logger.add('test', 'clean_acc', test_acc * 100, epoch)
    if epoch % args.adv_eval_freq == 0 or epoch == NUM_ADV_EPOCHS:
        test_adv_acc = trainer.eval(test_dataloader, adversarial=True, verbose=True)
        logger.add("test", "adversarial_acc", test_adv_acc * 100, epoch)
    end_ = time.time()
    logger.add('time', 'eval', format_time(end_ - start_), epoch)
    trainer.save_model(os.path.join(LOG_DIR, 'state-last.pt'), epoch)
    logger.log_info(epoch, ['train', 'test', 'eval', 'scheduler', 'time'])
    logger.plot_learning_curve()
    logger.save_stats('stats.pkl')

logger.log('\nTraining completed.')
if eval_dataloader:
    trainer.load_model(WEIGHTS)
    old_score[2] = trainer.eval(test_dataloader, adversarial=True, verbose=True)
    logger.log('Best checkpoint:  epoch-{}  test-nat-{:.2f}%  eval-adv-{:.2f}%  test-adv-{:.2f}%.'.format(best_epoch, old_score[0]*100, old_score[1]*100, old_score[2]*100))
logger.log('Last checkpoint:  epoch-{}  test-nat-{:.2f}%  test-adv-{:.2f}%.'.format(NUM_ADV_EPOCHS, test_acc*100, test_adv_acc*100))


