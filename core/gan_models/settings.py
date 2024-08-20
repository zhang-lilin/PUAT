import argparse


def parser_gan():
    parser = argparse.ArgumentParser(description='GD GAN.')
    # parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training.')
    # parser.add_argument('--data-dir', type=str, default='/cluster/home/rarade/data/')
    # parser.add_argument('--log-dir', type=str, default='/cluster/scratch/rarade/test/')
    
    parser.add_argument('-d', '--data', type=str, default='cifar10s', choices=DATASETS, help='Data to use.')
    parser.add_argument('--desc', type=str, required=True, 
                        help='Description of experiment. It will be used to name directories.')

    parser.add_argument('-m', '--model', choices=MODELS, default='wrn-28-10-swish', help='Model architecture to be used.')
    parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input.')
    parser.add_argument('--pretrained-file', type=str, default=None, help='Pretrained weights file name.')

    parser.add_argument('-na', '--num-adv-epochs', type=int, default=400, help='Number of adversarial training epochs.')
    parser.add_argument('--adv-eval-freq', type=int, default=25, help='Adversarial evaluation frequency (in epochs).')
    
    parser.add_argument('--beta', default=None, type=float, help='Stability regularization, i.e., 1/lambda in TRADES.')
    
    parser.add_argument('--lr', type=float, default=0.4, help='Learning rate for optimizer (SGD).')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Optimizer (SGD) weight decay.')
    parser.add_argument('--scheduler', choices=SCHEDULERS, default='cosinew', help='Type of scheduler.')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='Use Nesterov momentum.')
    parser.add_argument('--clip-grad', type=float, default=None, help='Gradient norm clipping.')

    parser.add_argument('-a', '--attack', type=str, choices=ATTACKS, default='linf-pgd', help='Type of attack.')
    parser.add_argument('--attack-eps', type=str2float, default=8/255, help='Epsilon for the attack.')
    parser.add_argument('--attack-step', type=str2float, default=2/255, help='Step size for PGD attack.')
    parser.add_argument('--attack-iter', type=int, default=10, help='Max. number of iterations (if any) for the attack.')
    parser.add_argument('--keep-clean', type=str2bool, default=False, help='Use clean samples during adversarial training.')

    parser.add_argument('--debug', action='store_true', default=False, 
                        help='Debug code. Run 1 epoch of training and evaluation.')
    
    parser.add_argument('--unsup-fraction', type=float, default=0.7, help='Ratio of unlabelled data to labelled data.')
    parser.add_argument('--aux-data-filename', type=str, help='Path to additional Tiny Images data.', 
                        default='/cluster/scratch/rarade/cifar10s/ti_500K_pseudo_labeled.pickle')
    
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')

    ### Consistency
    parser.add_argument('--consistency', action='store_true', default=False, help='use Consistency.')
    parser.add_argument('--cons_lambda', type=float, default=1.0, help='lambda for Consistency.')
    parser.add_argument('--cons_tem', type=float, default=0.5, help='temperature for Consistency.')

    ### Resume
    parser.add_argument('--resume_path', default='', type=str)

    ### Our methods
    parser.add_argument('--LSE', action='store_true', default=False, help='LSE training.')
    parser.add_argument('--ls', type=float, default=0.1, help='label smoothing.')
    parser.add_argument('--clip_value', default=0, type=float)
    parser.add_argument('--CutMix', action='store_true', default=False, help='use CutMix.')

    parser.add_argument('--mart', action='store_true', default=False, help='MART training.')
    parser.add_argument('--tardes', action='store_true', default=False, help='TRADES training.')
    parser.add_argument('--puat', action='store_true', default=False, help='PUAT training.')

    return parser


