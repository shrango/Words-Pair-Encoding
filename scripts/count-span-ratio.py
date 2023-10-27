import json
from collections import defaultdict
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--originfile')
parser.add_argument('--outputfile')
args = parser.parse_args()
with open(args.originfile,'r',encoding='utf-8')as f:
    content = f.readlines()
count = sum([len(line.split(' ')) for line in content])
print('There are {} words in total'.format(count))
with open(args.outputfile,'r')as f:
    content = json.load(f)
dic = defaultdict(int)
for k in content:
    for span in content[k]:
        dic[int(span[1])-int(span[0])]+=1
order = {k:dic[k] for k in sorted(dic)}
print("The span number",order)
spans = sum([k*dic[k] for k in dic])
ratio = spans/count
print("There are {} words counted as phrases, the phrase ratio is {}".format(spans,ratio))
