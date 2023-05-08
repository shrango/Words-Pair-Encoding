import argparse
import json
from tqdm import tqdm
from collections import defaultdict
parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--threshold',type=int,default=10)
parser.add_argument('--output')
args = parser.parse_args()
with open(args.input,'r')as f:
    content = json.load(f)
spanlen = defaultdict(int)
new_content = {}
for k in tqdm(content):
    new_content[k]=[]
    for span in content[k]:
        spanlen[span[1]-span[0]]+=1
        if span[1]-span[0] < args.threshold:
            new_content[k].append(span)
print(spanlen)
with open(args.output,'w')as f:
    json.dump(new_content,f)