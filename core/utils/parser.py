import argparse
import yaml
from core.attacks import ATTACKS
from core.data import DATASETS
from core.models import MODELS
from .train import SCHEDULERS
from .utils import str2bool, str2float



def set_parser_train(parser):
    parser.description = 'Standard + Adversarial Training.'

    parser.add_argument('--augment', type=str, default='base',
                        choices=['none', 'base'],
                        help='Augment training set.')

    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--batch-size-validation', type=int, default=512, help='Batch size for testing.')

    parser.add_argument('--data-dir', type=str, default='/data/')
    parser.add_argument('--log-dir', type=str, default='/test/')

    parser.add_argument('-d', '--data', type=str, default='cifar10', choices=DATASETS, help='Data to use.')
    parser.add_argument('--desc', type=str,
                        # required=True,
                        help='Description of experiment. It will be used to name directories.')
    parser.add_argument('--take_amount', type=int, default=None, help='Amount of data selected from base dataset.')

    parser.add_argument('-m', '--model', choices=MODELS, default='wrn-28-10-swish',
                        help='Model architecture to be used.')
    parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input.')
    parser.add_argument('--pretrained-file', type=str, default=None, help='Pretrained weights file name.')

    parser.add_argument('-na', '--num-adv-epochs', type=int, default=100, help='Number of adversarial training epochs.')
    parser.add_argument('--adv-eval-freq', type=int, default=25, help='Adversarial evaluation frequency (in epochs).')

    parser.add_argument('--lr', type=float, default=0.4, help='Learning rate for optimizer (SGD).')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Optimizer (SGD) weight decay.')
    parser.add_argument('--scheduler', choices=SCHEDULERS, default='cosinew', help='Type of scheduler.')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='Use Nesterov momentum.')
    parser.add_argument('--clip-grad', type=float, default=None, help='Gradient norm clipping.')

    parser.add_argument('-a', '--attack', type=str, choices=ATTACKS, default='linf-pgd', help='Type of attack.')
    parser.add_argument('--attack-eps', type=str2float, default=8 / 255, help='Epsilon for the attack.')
    parser.add_argument('--attack-step', type=str2float, default=2 / 255, help='Step size for PGD attack.')
    parser.add_argument('--attack-iter', type=int, default=10,
                        help='Max. number of iterations (if any) for the attack.')
    parser.add_argument('--keep-clean', type=str2bool, default=False,
                        help='Use clean samples during adversarial training.')

    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug code. Run 1 epoch of training and evaluation.')

    parser.add_argument('--unsup-fraction', type=float, default=0.7, help='Ratio of unlabelled data to labelled data.')
    parser.add_argument('--aux-data-filename', type=str, help='Path to additional data.',
                        default=None)
    parser.add_argument('--aux_take_amount', type=int, default=None,
                        help='Amount of data selected from unlabelled dataset.')

    parser.add_argument('-seed', '--seed', type=int, default=1, help='Random seed.')

    ### Consistency
    parser.add_argument('--consistency', action='store_true', default=False, help='use Consistency.')
    parser.add_argument('--cons_lambda', type=float, default=1.0, help='lambda for Consistency.')
    parser.add_argument('--cons_tem', type=float, default=0.5, help='temperature for Consistency.')

    ### Resume
    parser.add_argument('--resume_path', default='', type=str)
    parser.add_argument('--pre_resume_path', default='', type=str)

    ### Our methods
    parser.add_argument('--LSE', action='store_true', default=False, help='LSE training.')
    parser.add_argument('--ls', type=float, default=0.1, help='label smoothing.')
    parser.add_argument('--clip_value', default=0, type=float)
    parser.add_argument('--CutMix', action='store_true', default=False, help='use CutMix.')

    parser.add_argument('--shuffle_train', action='store_true', default=True)

    parser.add_argument('--puat', action='store_true', default=False, help='PUAT training.')
    parser.add_argument('--beta', default=None, type=float, help='Weight of the loss over UAE.')
    parser.add_argument('--beta2', default=None, type=float, help='Weight of the loss over RAE.')

    return parser


def set_parser_eval(parser):
    parser.description = 'Robustness evaluation.'

    parser.add_argument('-d', '--data', type=str, default='cifar10s', choices=DATASETS, help='Data to use.')
    parser.add_argument('-desc', '--desc', type=str,
                        # required=True,
                        help='Description of experiment. It will be used to name directories.')
    parser.add_argument('-seed',"--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "-attack-eps", "--attack-eps", type=str2float, default=8 / 255, help="Epsilon for the attack."
    )
    parser.add_argument(
        "--attack-step", type=str2float, default=1 / 255, help="Step size for attack."
    )
    parser.add_argument(
        "--attack-iter",
        type=int,
        default=20,
        help="Max. number of iterations (if any) for the attack.",
    )

    parser.add_argument(
        "-batch-size-validation", "--batch-size-validation", type=int, default=512, help="Batch size for testing."
    )
    parser.add_argument("-resume_path", "--resume_path", default="", type=str)
    parser.add_argument("-save", "--save", default=False, type=bool)

    return parser