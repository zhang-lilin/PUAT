import logging
import os
import pickle
import torchvision
from matplotlib import pyplot as plt
import coloredlogs

def get_logger(logger_name=None):
    if logger_name is not None:
        logger = logging.getLogger(logger_name)
        logger.propagate = 0
    else:
        logger = logging.getLogger("taufikxu")
    return logger


def build_logger(path, logger_name=None):
    # FORMAT = "%(asctime)s|%(levelname)s%(message)s"
    FORMAT = "%(message)s"
    DATEF = "%H-%M-%S"
    logging.basicConfig(format=FORMAT)
    logger = get_logger(logger_name)

    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s;%(levelname)s|%(message)s", "%H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    LEVEL_STYLES = dict(
        debug=dict(color="magenta"),
        info=dict(color="green"),
        verbose=dict(),
        warning=dict(color="blue"),
        error=dict(color="yellow"),
        critical=dict(color="red", bold=True),
    )
    coloredlogs.install(
        level=logging.INFO, fmt=FORMAT, datefmt=DATEF, level_styles=LEVEL_STYLES
    )
    return logger


class Logger(object):
    """
    Helper class for logging.
    Arguments:
        path (str): Path to log file.
    """
    def __init__(self, path):
        # self.logger = logging.getLogger()
        self.path = path
        self.logger = build_logger(path)
        self.log_dir = os.path.dirname(path)
        # self.setup_file_logger()
        self.stats = dict()
        print ('Logging to file: ', self.path)
        
    # def setup_file_logger(self):
    #     hdlr = logging.FileHandler(self.path, 'w+')
    #     self.logger.addHandler(hdlr)
    #     self.logger.setLevel(logging.INFO)

    def log(self, message):
        # print (message)
        self.logger.info(message)

    def log_info(self, step, cats=None):
        if cats is None:
            cats = self.stats.keys()
        text = ''
        for cat in cats:
            if cat in self.stats:
                text += "|{}: ".format(cat)
                for k in self.stats[cat]:
                    last_v = self.stats[cat][k][-1]
                    if last_v[0] == step:
                        if isinstance(last_v[1], float):
                            form = "{}:{:.2f}% " if 'acc' in k else "{}:{:.5f} "
                            text += form.format(k, last_v[1])
                        else:
                            text += "{}:{} ".format(k, last_v[1])
                text += "\n"
        if len(text) > 0:
            self.logger.info(text)

    def add(self, category, k, v, global_it, unique=False):
        if category not in self.stats:
            self.stats[category] = {}
        if k not in self.stats[category]:
            self.stats[category][k] = []
        if unique:
            for tup in self.stats[category][k]:
                if tup[0] == global_it:
                    self.stats[category][k].remove(tup)
        self.stats[category][k].append((global_it, v))

    def save_stats(self, filename=None):
        if filename is None:
            filename = "stat.pkl"
        filename = os.path.join(self.log_dir, filename)

        with open(filename, mode='wb') as f:
            pickle.dump(self.stats, f)

    def save_images(self, imgs, name=None, class_name=None, nrow=10):
        if class_name is None:
            outdir = self.log_dir
        else:
            outdir = os.path.join(self.log_dir, class_name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if isinstance(name, str):
            outfile = os.path.join(outdir, "{}.png".format(name))
        else:
            outfile = os.path.join(outdir, "%08d.png" % name)

        torchvision.utils.save_image(imgs, outfile, nrow=nrow)

    def plot_learning_curve(self, test=True, validation=True):
        stat = self.stats
        cats = ['train']
        if test:
            cats.append("test")
        if validation:
            cats.append("eval")
        assert len(cats) > 0

        figure = plt.figure()
        for i in cats:
            try:
                dat = stat[i]["clean_acc"]
            except:
                dat = None
            finally:
                if dat is not None:
                    iter_list, acc_list = [], []
                    for tup in dat:
                        iter_list.append(tup[0])
                        acc_list.append(tup[1])
                    plt.plot(iter_list, acc_list, "-", label="{}-nat".format(i))
            try:
                rob = stat[i]["adversarial_acc"]
            except:
                rob = None
            finally:
                if rob is not None:
                    iter_list1, acc_list1 = [], []
                    for tup in rob:
                        iter_list1.append(tup[0])
                        acc_list1.append(tup[1])
                    plt.plot(iter_list1, acc_list1, "-", label="{}-rob".format(i))

        plt.legend()
        plt.grid()
        filename = os.path.join(self.log_dir, "acc.png")
        figure.savefig(filename)
        plt.cla()
        plt.close(figure)

    def plot_loss(self, cats, filename):
        stat = self.stats
        figure = plt.figure()
        for i in cats:
            try:
                dat = stat[i]['loss']
            except:
                dat = None
            finally:
                if dat is not None:
                    iter_list, acc_list = [], []
                    for tup in dat:
                        iter_list.append(tup[0])
                        acc_list.append(tup[1])
                    plt.plot(iter_list, acc_list, "-", label=i)

        plt.legend()
        plt.grid()
        filename = os.path.join(self.log_dir, "{}.png".format(filename))
        figure.savefig(filename)
        plt.cla()
        plt.close(figure)

    def plot_loss_gan(self):
        cats = ['training_d', 'training_g']
        self.plot_loss(cats, 'gan')
        cats = ['training', 'training_a']
        self.plot_loss(cats, 'adv')

