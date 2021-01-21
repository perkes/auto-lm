from subprocess import Popen
import json

experiments = json.load(open('./data/linear_experiments.json', 'r'))

for params in experiments:
    p = Popen('python experiment.py --params "' + str(params) + '"', shell = True)
    ret = p.wait()
