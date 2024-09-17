import argparse
from utils.util import get_logger
import os
import yaml
def read_name(config_path):
    nameList = config_path.split('/')
    allName = nameList[-1]
    nameList = allName.split('.')
    return nameList[0]
def genPath(path_str, date):
    return path_str+date+'/'
def print_args(args):
    log_level = 'INFO'
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(args.logdir, __name__, 'info_{}.log'.format(args.name), level=log_level)
    logger.info(args)
    # logger.info(args.dist_thr)

    return logger


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)
def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
def dict2args(config, args):
    keys = config.keys()
    for key in keys:
        for k, v in config[key].items(): 
            setattr(args, k, v)
        
    return args
def parse_args():
    parser = argparse.ArgumentParser(description='DiffSTG@jbtang')
    parser.add_argument('--config', default='./config/teacher_gwnet.yaml', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as y:
        config = yaml.safe_load(y)
    # config.update({'name': read_name(args.config)})
    args = dict2args(config, args)
    args.mdir = genPath(args.mdir, args.date)
    args.logdir = genPath(args.logdir, args.date)
    args.name = read_name(args.config)

    return args
    
args = parse_args()
logger = print_args(args)