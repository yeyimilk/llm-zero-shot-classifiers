import argparse


class DArgs:
    def __init__(self, ds=0, prompt_type='few_shot') -> None:
        self.ds = ds
        self.prompt_type = prompt_type
class Config:
    def __init__(self) -> None:
        self.args = DArgs()

cfig = Config()

def run_args():
    parser = argparse.ArgumentParser(description='Process parameters.')
    parser.add_argument('--ds', type=int, default=0, help='dataset indexing')
    parser.add_argument('--prompt_type', type=str, default='few_shot', help='basic, few_shot')
    cfig.args = parser.parse_args()