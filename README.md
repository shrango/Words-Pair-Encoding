<<<<<<< HEAD
This is WPE
=======
# Words-Pair-Encoding
let's assume the very begining data are called "train.raw.src", "valid.raw.src"(optional), "test.raw.src"(optional)
process should be done before WPE:
1. Tokenization. And you'll get "train.tok.src"
2. Traditional BPE. And you'll get "train.tok.bpe.src", as well as "bpe.code" and "vocab.src"(optional)
3. WPE with "train.tok.src", "bpe.code" and "vocab.src"(optional). And you'll get "phrase.some-long-middle-name.src" and "final.train.recover.src"
4. (Optional) Compare "train.tok.bpe.src" and "final.train.recover.src" by `ls -ll` to make sure 100% recovery.

# How to use WPE?
1. Copy your data into `Data` file, this should include "train.tok.src", "bpe.code". "valid.tok.src", "test.tok.src", and "vocab.src" are optional at WPE training stage.
2. Refill `inputtrainfile`, `inputfile`, `wordcode` and `vocab.src`(optional) in `oneline-getphrasespans.sh` according to your files' name.
3. Just run `bash oneline-getphrasespans.sh`.
>>>>>>> origin/main
