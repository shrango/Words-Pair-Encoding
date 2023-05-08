#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
Unlike the original BPE, it does not compress the plain text, but can be used to reduce the vocabulary
of a text to a configurable number of symbols, with only a small increase in the number of tokens.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals
# python learn_bpe.py --input test.en -s 1000 --output temp.out --num-workers 1 --sep #$&
import os
import sys
import inspect
import codecs
import re
import copy
import argparse
import warnings
import tempfile
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter
import pdb
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator

# hack for python2/3 compatibility
from io import open
argparse.open = open

def create_parser(subparsers=None):

    if subparsers:
        parser = subparsers.add_parser('learn-bpe',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")
    else:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input text (default: standard input).")

    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s)")
    parser.add_argument(
        '--min-frequency', type=float, default=1e-8, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s)')
    parser.add_argument('--dict-input', action="store_true",
        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument(
        '--total-symbols', '-t', action="store_true",
        help="subtract number of characters from the symbols to be generated (so that '--symbols' becomes an estimate for the total number of symbols needed to encode text).")
    parser.add_argument(
        '--num-workers', type=int, default=1,
        help="Number of processors to process texts, only supported in Python3. If -1, set `multiprocessing.cpu_count()`. (default: %(default)s)")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")
    parser.add_argument(
        '--sep', type=str, default='',
        help='两个char/token之间的分隔符'
    )
    parser.add_argument(
        '--delta', type=int, default=10,
        help='超参'
    )
    parser.add_argument(
        '--scorefuc', type=str, default='sqrt',
        help='代表打分函数类型，sqrt表示(w1,w2-δ)/sqrt{w1*w2}，min表示(w1,w2-δ)/min{w1,w2}'
    )

    return parser
sep_sign = ''
delta = 10
score_fun = 'sqrt'
global_uni_stats={}
global_pair_freq={}
def get_vocabulary(fobj, is_dict=False, num_workers=1):
    """Read text and return dictionary that encodes vocabulary
    """
    vocab = Counter()
    if is_dict:
        for i, line in enumerate(fobj):
            try:
                word, count = line.strip('\r\n ').split(' ')
            except:
                print('Failed reading vocabulary file at line {0}: {1}'.format(i, line))
                sys.exit(1)
            vocab[word] += int(count)
    elif num_workers == 1 or fobj.name == '<stdin>':
        if num_workers > 1:
            warnings.warn("In parallel mode, the input cannot be STDIN. Using 1 processor instead.")
        for i, line in enumerate(fobj):
            for word in line.strip('\r\n ').split(' '):
                if word:
                    vocab[word] += 1
    elif num_workers > 1:

        if sys.version_info < (3, 0):
            print("Parallel mode is only supported in Python3.")
            sys.exit(1)

        with open(fobj.name, encoding="utf8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = int(size / num_workers)
            offsets = [0 for _ in range(num_workers + 1)]
            for i in range(1, num_workers):
                f.seek(chunk_size * i)
                pos = f.tell()
                while True:
                    try:
                        line = f.readline()
                        break
                    except UnicodeDecodeError:
                        pos -= 1
                        f.seek(pos)
                offsets[i] = f.tell()
                assert 0 <= offsets[i] < 1e20, "Bad new line separator, e.g. '\\r'"

        vocab_files = []
        pool = Pool(processes=num_workers)
        for i in range(num_workers):
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.close()
            vocab_files.append(tmp)
            pool.apply_async(_get_vocabulary, (fobj.name, tmp.name, offsets[i], offsets[i + 1]))
        pool.close()
        pool.join()
        import pickle
        for i in range(num_workers):
            with open(vocab_files[i].name, 'rb') as f:
                vocab += pickle.load(f)
            os.remove(vocab_files[i].name)
    else:
        raise ValueError('`num_workers` is expected to be a positive number, but got {}.'.format(num_workers))
    return vocab

def _get_vocabulary(infile, outfile, begin, end):
    import pickle
    vocab = Counter()
    with open(infile, encoding="utf8") as f:
        f.seek(begin)
        line = f.readline()
        while line:
            pos = f.tell()
            assert 0 <= pos < 1e20, "Bad new line separator, e.g. '\\r'"
            if end > 0 and pos > end:
                break
            for word in line.strip('\r\n ').split(' '):
                if word:
                    vocab[word] += 1
            line = f.readline()
    with open(outfile, 'wb') as f:
        pickle.dump(vocab, f)

def update_pair_statistics(pair, changed, stats, indices):
    # 更新stats的时候不能简单的减一，因为里面不是简单的频率了，而是要重新按score_func算
    """Minimally update the indices and frequency of symbol pairs

    if we merge a pair of symbols, only pairs that overlap with occurrences
    of this pair are affected, and need to be updated.
    """
    # pdb.set_trace()
    # global的写法很答辩，但是它没有封装成一个类，我也没办法啊
    global global_uni_stats
    global global_pair_freq
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first+sep_sign+second
    # pdb.set_trace()
    for j, word, old_word, freq in changed:
        need_update = set([])
        # find all instances of pair, and update frequency/indices around it
        i = 0
        while True:
            # find first symbol
            try:
                i = old_word.index(first, i)
            except ValueError:
                break
            # if first symbol is followed by second symbol, we've found an occurrence of pair (old_word[i:i+2])
            if i < len(old_word)-1 and old_word[i+1] == second:
                # assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
                # pdb.set_trace()
                if i:
                    prev = old_word[i-1:i+1]
                    # if prev[0]=='narrowness' or prev[1]=='narrowness':
                    #     print("A")
                    #     pdb.set_trace()
                    global_uni_stats[prev[0]]-=freq
                    global_uni_stats[prev[1]]-=freq
                    global_pair_freq[prev]-=freq
                    if score_fun in ('sqrt','min'):
                        stats[prev] = -100
                    else:
                        stats[prev] -= freq
                    indices[prev][j] -= 1
                if i < len(old_word)-2:
                    # assuming a symbol sequence "A B C B", if "B C" is merged, reduce the frequency of "C B".
                    # however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced by the previous code block
                    if old_word[i+2] != first or i >= len(old_word)-3 or old_word[i+3] != second:
                        nex = old_word[i+1:i+3]
                        # if nex[0]=='narrowness' or nex[1]=='narrowness':
                        #     print("B")
                        #     pdb.set_trace()
                        # 在我的情况下还得考虑"A B C D B C"的情况下，D的uni频率被减2次的问题
                        if i<len(old_word)-4 and old_word[i+3:i+5]==pair:
                            pass
                        else:
                            global_uni_stats[nex[0]]-=freq
                            global_uni_stats[nex[1]]-=freq
                        global_pair_freq[nex]-=freq
                        # 减的话一定会减完，所以一定是0，那么打分取一个很小的数就行了，不用再算
                        if score_fun in ('sqrt','min'):
                            stats[nex] = -100
                        else:
                            stats[nex] -= freq
                        indices[nex][j] -= 1
                i += 2
            else:
                i += 1

        i = 0
        while True:
            # pdb.set_trace()
            try:
                # find new pair
                i = word.index(new_pair, i)
            except ValueError:
                break
            # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
            if i:
                prev = word[i-1:i+1]
                global_uni_stats[prev[0]]+=freq
                global_uni_stats[prev[1]]+=freq
                global_pair_freq[prev]+=freq
                need_update.add(prev)
                # stats[prev] += freq
                # if score_fun=='sqrt':
                #     stats[prev] = (global_pair_freq[prev]-1-delta)/(global_uni_stats[prev[0]]*global_uni_stats[prev[1]])**0.5
                # elif score_fun=='min':
                #     print("还没实现，来line249找")
                indices[prev][j] += 1
            # assuming a symbol sequence "A BC B", if "B C" is merged, increase the frequency of "BC B"
            # however, if the sequence is A BC BC, skip this step because the count of "BC BC" will be incremented by the previous code block
            if i < len(word)-1 and word[i+1] != new_pair:
                nex = word[i:i+2]
                # if nex[1]=='narrowness':
                #     pdb.set_trace()
                global_uni_stats[nex[0]]+=freq
                global_uni_stats[nex[1]]+=freq
                if not global_uni_stats[nex[0]]*global_uni_stats[nex[1]]:
                    print("警告！出现了问题,",word,'问题词是',nex)
                    if not global_uni_stats[nex[0]]:
                        global_uni_stats[nex[0]]=1
                    else:
                        global_uni_stats[nex[1]]=1
                    # pdb.set_trace()
                global_pair_freq[nex]+=freq
                need_update.add(nex)
                # if score_fun=='sqrt':
                #     stats[nex] = (global_pair_freq[nex]-1-delta)/(global_uni_stats[nex[0]]*global_uni_stats[nex[1]])**0.5
                # elif score_fun=='min':
                #     print("还没实现，来line261找")
                # stats[nex] += freq
                indices[nex][j] += 1
            i += 1
        for p in need_update:
            stats[p] = (global_pair_freq[p]-1-delta)/(global_uni_stats[p[0]]*global_uni_stats[p[1]])**0.5


def get_pair_statistics(vocab):
    """Count frequency of all symbol pairs, and create index"""

    # data structure of pair frequencies
    stats = defaultdict(int)
    uni_stats = defaultdict(int)
    #index from pairs to words
    indices = defaultdict(lambda: defaultdict(int))
    # pdb.set_trace()
    for i, (word, freq) in enumerate(vocab):
        prev_char = word[0]
        uni_stats[prev_char] += freq
        for char in word[1:]:
            uni_stats[char] += freq
            stats[prev_char, char] += freq
            indices[prev_char, char][i] += 1
            prev_char = char

    return stats, indices, uni_stats


def replace_pair(pair, vocab, indices):
    """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
    first, second = pair
    pair_str = sep_sign.join(pair)
    pair_str = pair_str.replace('\\','\\\\')
    changes = []
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
    if sys.version_info < (3, 0):
        iterator = indices[pair].iteritems()
    else:
        iterator = indices[pair].items()
    for j, freq in iterator:
        if freq < 1:
            continue
        word, freq = vocab[j]
        new_word = ' '.join(word)
        new_word = pattern.sub(pair_str, new_word)
        new_word = tuple(new_word.split(' '))

        vocab[j] = (new_word, freq)
        changes.append((j, new_word, word, freq))

    return changes

def prune_stats(stats, big_stats, threshold):
    """Prune statistics dict for efficiency of max()

    The frequency of a symbol pair never increases, so pruning is generally safe
    (until we the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    for item,freq in list(stats.items()):
        if freq < threshold:
            del stats[item]
            if freq < 0:
                # big_stats[item] += freq
                big_stats[item] = -1
            else:
                big_stats[item] = freq


def learn_bpe(infile, outfile, num_symbols, min_frequency=1e-8, verbose=False, is_dict=False, total_symbols=False, num_workers=1):
    """Learn num_symbols BPE operations from vocabulary, and write to outfile.
    """

    # version 0.2 changes the handling of the end-of-word token ('</w>');
    # version numbering allows bckward compatibility
    outfile.write('#version: 0.2\n')
    # pdb.set_trace()
    # vocab = get_vocabulary(infile, is_dict, num_workers)
    content = infile.readlines()
    vocab = defaultdict(int)
    for line in content:
        k = tuple(line.replace('\n','</w>').split(' '))
        vocab[k]+=1
    # pdb.set_trace()
    # vocab = dict([(tuple(x[:-1])+(x[-1]+'</w>',) ,y) for (x,y) in vocab.items()])
    print("delta是",delta)
    print("sep是",sep_sign)
    print("最小分数",min_frequency)
    print("打分函数是",score_fun)
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    global global_uni_stats
    global global_pair_freq
    stats, indices, global_uni_stats = get_pair_statistics(sorted_vocab)
    global_pair_freq = stats
    if score_fun=="sqrt":
        score_stats = {pair:(global_pair_freq[pair]-delta)/(global_uni_stats[pair[0]]*global_uni_stats[pair[1]])**0.5 for pair in stats}
    elif score_fun=="min":
        score_stats = {pair:(global_pair_freq[pair]-delta)/min(global_uni_stats[pair[0]],global_uni_stats[pair[1]]) for pair in stats}
    else:
        assert False, "score_function错误"
        
    # pdb.set_trace()# 47124
    big_stats = copy.deepcopy(score_stats)

    if total_symbols:
        uniq_char_internal = set()
        uniq_char_final = set()
        for word in vocab:
            for char in word[:-1]:
                uniq_char_internal.add(char)
            uniq_char_final.add(word[-1])
        sys.stderr.write('Number of word-internal characters: {0}\n'.format(len(uniq_char_internal)))
        sys.stderr.write('Number of word-final characters: {0}\n'.format(len(uniq_char_final)))
        sys.stderr.write('Reducing number of merge operations by {0}\n'.format(len(uniq_char_internal) + len(uniq_char_final)))
        num_symbols -= len(uniq_char_internal) + len(uniq_char_final)

    # threshold is inspired by Zipfian assumption, but should only affect speed
    threshold = max(score_stats.values()) / 20000
    # enzh大数据集上的阈值稍微调低一些
    # pdb.set_trace()
    print("阈值是",threshold)
    for i in tqdm(range(num_symbols)):
        # pdb.set_trace()
        if score_stats:
            most_frequent = max(score_stats, key=lambda x: (score_stats[x], x))
        # print(most_frequent)
        # we probably missed the best pair because of pruning; go back to full statistics
        if not score_stats or (i and score_stats[most_frequent] < threshold):
            prune_stats(score_stats, big_stats, threshold)
            score_stats = copy.deepcopy(big_stats)
            most_frequent = max(score_stats, key=lambda x: (score_stats[x], x))
            # threshold is inspired by Zipfian assumption, but should only affect speed
            threshold = score_stats[most_frequent] * i/(i+10000.0)
            prune_stats(score_stats, big_stats, threshold)

        if score_stats[most_frequent] < min_frequency:
            sys.stderr.write('no pair has frequency >= {0}. Stopping\n'.format(min_frequency))
            break

        if verbose:
            sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, most_frequent[0], most_frequent[1], stats[most_frequent]))
        # pdb.set_trace()
        outfile.write('{0} {1}\n'.format(*most_frequent))
        # outfile.write(most_frequent[0]+sep_sign+most_frequent[1]+'\n')
        # pdb.set_trace()
        changes = replace_pair(most_frequent, sorted_vocab, indices)
        update_pair_statistics(most_frequent, changes, score_stats, indices)
        score_stats[most_frequent] = 0
        if not i % 100:
            prune_stats(score_stats, big_stats, threshold)


if __name__ == '__main__':

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    newdir = os.path.join(currentdir, 'subword_nmt')
    if os.path.isdir(newdir):
        warnings.warn(
            "this script's location has moved to {0}. This symbolic link will be removed in a future version. Please point to the new location, or install the package and use the command 'subword-nmt'".format(newdir),
            DeprecationWarning
        )

    # python 2/3 compatibility
    # if sys.version_info < (3, 0):
    #     sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
    #     sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
    #     sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    # else:
    #     sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
    #     sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
    #     sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    parser = create_parser()
    args = parser.parse_args()
    sep_sign = args.sep
    delta = args.delta
    score_fun = args.scorefuc
    if args.num_workers <= 0:
        args.num_workers = cpu_count()

    if sys.version_info < (3, 0) and args.num_workers > 1:
        args.num_workers = 1
        warnings.warn("Parallel mode is only supported in Python3. Using 1 processor instead.")

    # read/write files as UTF-8
    # if args.input.name != '<stdin>':
    #     args.input = codecs.open(args.input.name, encoding='utf-8')
    # if args.output.name != '<stdout>':
    #     args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    args.input = codecs.open(args.input.name, encoding='utf-8')
    args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    learn_bpe(args.input, args.output, args.symbols, args.min_frequency, args.verbose, is_dict=args.dict_input, total_symbols=args.total_symbols, num_workers=args.num_workers)
    args.input.close()
    args.output.close()
    # close files
    # if args.input.name != '<stdin>':
    #     args.input.close()
    # if args.output.name != '<stdout>':
    #     args.output.close()
