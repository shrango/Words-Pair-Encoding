This is WPE
=======
WPE is used to extract semantic units(you may simply treat them as phrases), using statistical self-supervised method.

For more introduction, please refer to my paper: **Enhancing Neural Machine Translation with Semantic Units**, EMNLP 2023(findings).

Note: This project is a rough version, yet usable. Anyway, I will complete it soon.
=======
I used `subword-nmt` to carry out the BPE operation. Please make sure you have installed [`subword-nmt`](https://github.com/rsennrich/subword-nmt) on your system.

# Words-Pair-Encoding
let's assume the very begining data are called "train.raw.src", "valid.raw.src"(optional), "test.raw.src"(optional)

process should be done before WPE:
1. Tokenization. And you'll get "train.tok.src"
2. Traditional BPE(using [`subword-nmt`](https://github.com/rsennrich/subword-nmt)). And you'll get "train.tok.bpe.src", as well as "bpe.code" and "vocab.src"(optional)
3. WPE with "train.tok.src", "bpe.code" and "vocab.src"(optional). And you'll get "phrase.some-long-middle-name.src" and "final.train.recover.src"
4. (Optional) Compare "train.tok.bpe.src" and "final.train.recover.src" by `ls -ll` to make sure 100% recovery. If they are different in file size, try to add/remove "--vocabulary" in "subword-nmt apply-bpe" operation, which I have noted in `oneline-getphrasespans.sh` file.

# How to run WPE?
1. Copy your data into `Data` file, this should include "train.tok.src", "bpe.code". "valid.tok.src", "test.tok.src", and "vocab.src" are optional at WPE training stage.
2. Refill `inputtrainfile`, `inputfile`, `wordcode` and `vocab.src`(optional) in `oneline-getphrasespans.sh` according to your files' name.
3. Just run `bash oneline-getphrasespans.sh`.
