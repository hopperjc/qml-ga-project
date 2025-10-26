import argparse
from experiments.runner import ExperimentRunner

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default="configs/experiment.yaml")
    args = p.parse_args()
    ExperimentRunner().run_from_yaml(args.config)
