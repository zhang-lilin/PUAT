import os
import multiprocessing
import itertools


def exp_runner(para):
    cmd, gpus, proc_to_gpu_map = para[:1][0], para[1:2][0], para[2:3][0]
    process_id = multiprocessing.current_process().name
    if process_id not in proc_to_gpu_map:
        proc_to_gpu_map[process_id] = gpus.pop()
        print("assign gpu {} to {}".format(proc_to_gpu_map[process_id], process_id))
    gpuid = proc_to_gpu_map[process_id]
    # return os.system(cmd + " -gpu {} ".format(gpuid))
    return os.system("CUDA_VISIBLE_DEVICES={} ".format(gpuid) + cmd)
    # return os.system(cmd)


if __name__ == '__main__':

    gpu_list = [0]
    dataset = "cifar10"
    commands = []

    args_fortune = {
        "TUNNER_groups": [
            f"python train_wa_puat.py ./configs/puat_{dataset}_pre.yaml ",
            f"python train_wa_puat.py ./configs/puat_{dataset}.yaml ",
        ],

        "seed": [1, 2, 3],

    }

    command_template = ""
    key_sequence = []
    for k in args_fortune:
        key_sequence.append(k)
        if k == "config_file" or "TUNNER" in k:
            command_template += " {}"
        else:
            command_template += " -" + k + " {}"
    print(command_template, key_sequence)

    possible_value = []
    for k in key_sequence:
        possible_value.append(args_fortune[k])

    for args in itertools.product(*possible_value):
        commands.append(command_template.format(*args))
    print(commands)

    print("# experiments = {}".format(len(commands)))
    gpus = multiprocessing.Manager().list(gpu_list)
    proc_to_gpu_map = multiprocessing.Manager().dict()

    p = multiprocessing.Pool(processes=len(gpu_list))
    para = [[com,gpus,proc_to_gpu_map] for com in commands]
    rets = p.map(exp_runner, para)
    p.close()
    p.join()
    print(rets)

