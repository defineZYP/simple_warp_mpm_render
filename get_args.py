import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0)

    args = parser.parse_args()
    return args
