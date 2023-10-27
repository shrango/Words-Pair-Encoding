import re
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output',default=None)
args = parser.parse_args()
pat = re.compile('\w')
strange_pat = re.compile('&#\d*?')
sep_pat=re.compile(' |#\$&')
# python filter-non-character.py --input ende14.390w.delta100.1w.root.code --output character-phrase.1w.code
def keep(s):
    # w1, w2 = s.split(' ')
    # w2 = w2.replace('</w>','')
    # if re.findall(strange_pat,w1) or re.findall(strange_pat,w2):
    #     return False
    # if re.findall(pat,w1) and re.findall(pat,w2):
    #     return True
    # else:
    #     return False
    words = re.split(sep_pat, s)
    words[-1] = words[-1].replace('</w>','')
    for word in words:
        if re.findall(strange_pat, word):
            return False
    for word in words:
        if not re.findall(pat, word):
            return False
    return True

with open(args.input,'r',encoding='utf-8')as f:
    content = f.readlines()

remains = [line for line in content if keep(line)]
print('There are {} phrases before filter, and {} after'.format(len(content),len(remains)))
if not args.output:
    args.output = args.input+'.filterout'
with open(args.output,'w',encoding='utf-8')as f:
    for line in remains:
        f.write(line)
