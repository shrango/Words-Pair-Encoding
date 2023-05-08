import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output')
args = parser.parse_args()
# python remove_placeholder.py --input train.phrase1w.bpe32000.en --output train.phrase1w.bpe32000.clean.en
with open(args.input,'r',encoding='utf-8')as f:
    content = f.readlines()

content = [s.replace('#@@ $@@ & ','') for s in content]
with open(args.output,'w',encoding='utf-8')as f:
    for line in tqdm(content):
        f.write(line)