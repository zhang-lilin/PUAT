import itertools
import os


methods = [
    'puat-pre',
    'puat',
]

dataset = 'cifar10'
ARGS_FOR_TUNE = dict(
    seed = [1,2,3,],
)
if __name__ == '__main__':
    gpu = 0
    commands = []
    tunner_groups = []
    for method in methods:
        tunner_groups.append(f"python -X faulthandler train.py ./configs/{method}_{dataset}.yaml")

    command_template = " {}"
    for k in ARGS_FOR_TUNE:
        command_template += " --" + k + " {}"
    possible_value = []
    possible_value.append(tunner_groups)
    for k in ARGS_FOR_TUNE:
        possible_value.append(ARGS_FOR_TUNE[k])
    for args in itertools.product(*possible_value):
        commands.append(command_template.format(*args))
    print(commands)
    print("# experiments = {}".format(len(commands)))

    def exp_runner(com, gpu):
        return os.system("CUDA_VISIBLE_DEVICES={} ".format(gpu) + com)
    for com in commands:
        exp_runner(com, gpu)

