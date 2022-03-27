import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str)
parser.add_argument('--file-name', type=str)
args = parser.parse_args()

input_file = open(os.path.join(args.data_dir, "train."+args.file_name), "r")
vis = {}
for line in input_file:
    line = line.strip().split()
    if (len(line) == 0):
        continue
    vis[line[1]] = True

out_file = open(os.path.join(args.data_dir, "ner_labels.txt"), "w")
for key, _ in vis.items():
    out_file.write(key+'\n')