import os
import time
import subprocess

import argparse
# some CLI arguments
parser = argparse.ArgumentParser(description='Train / evaluate all downstream classifiers.')
parser.add_argument('--eval', default=False, action='store_true', help='eval mode only')
parser.add_argument('--retrain', default=False, action='store_true', help='force retrain')
args = parser.parse_args()

commands = []

# all targets
targets = ["movement", "movement_up", "anomaly", "future_anomaly", "positive"]

# iterate over all folders
for folder in os.listdir("logs/"):
    for target in targets:
        commands.append('python -O classify.py "%s" --target %s' % (os.path.join("logs", folder), target))

if args.eval:
    commands = [cmd + " --nobalance" for cmd in commands]

if args.retrain:
    commands = [cmd + " --retrain" for cmd in commands]

# all the commands to run
print("Going to run following commands (%d):" % (len(commands)))
for command in commands:
    print("-", command)

import parallel
parallel.run(commands, num=4)