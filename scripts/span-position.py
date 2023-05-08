import argparse
import json
from tqdm import tqdm
import re
parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output')
parser.add_argument('--withbpe',action='store_true')
args = parser.parse_args()
# python span-position.py --input train.phrase1w.bpe32000.en --out spans.phrase1w.bpe32000.en

with open(args.input,'r',encoding='utf-8')as f:
    content = f.readlines()
# content = [s.replace('#@@ $@@ &@@ ','#$& ') for s in content]
content = [s.replace(' #@@ $@@ & ','@@ #$& ') for s in content]

# print(''.join(content))
phrase_spans = {}
if args.withbpe:
    for idx, line in tqdm(enumerate(content)):
        phrase_spans[idx]=[]
        tokens = line.split(' ')
        count_len = 0
        in_phrase = False
        # length = len(tokens)
        phrase_len = 0
        pos = 0
        while pos < len(tokens):
            # A@@ #$& B@@ C@@ #$& D@@ E@@ F G H@@ #$& I J@@ K L
            # A@@ #$& B@@ C@@ #$& D@@ E@@ F G H@@ #$& I J@@ K@@ L
            if tokens[pos]=='#$&':
                in_phrase = True
            elif '@@' in tokens[pos]:
                in_phrase = True
                count_len+=1
                phrase_len+=1
            else:
                count_len+=1
                if in_phrase:
                    phrase_len+=1
                    phrase_spans[idx].append([count_len-phrase_len, count_len])
                phrase_len = 0
                in_phrase = False
            pos+=1
else:
    for idx, line in tqdm(enumerate(content)):
        phrase_spans[idx]=[]
        tokens = line.split(' ')
        count_len = 0
        in_phrase = False
        # length = len(tokens)
        phrase_len = 0
        pos = 0
        while pos < len(tokens):
            # A@@ #$& B@@ C@@ #$& D@@ E@@ F G H@@ #$& I J
            if tokens[pos]=='#$&':
                in_phrase = True
            elif '@@' in tokens[pos]:
                count_len+=1
                phrase_len+=1
            else:
                count_len+=1
                if in_phrase:
                    phrase_len+=1
                    phrase_spans[idx].append([count_len-phrase_len, count_len])
                phrase_len = 0
                in_phrase = False
            pos+=1

with open(args.output,'w')as f:
    json.dump(phrase_spans, f)