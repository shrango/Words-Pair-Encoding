# tokenize之后bpe之前
inputtrainfile=Data/train.tok.en
deltanum=100
phrasecode=tempfile/ende14.390w.delta$deltanum.2w.root.code
operation_num=20000
ratio_status=true

echo "开始学习短语bpe"
python scripts/learn_phrase_bpe_2.0.py --input $inputtrainfile -s $operation_num --output $phrasecode.temp --delta $deltanum --num-workers 1 --sep "#$&"
echo "短语bpe学习完成，进行过滤"
python scripts/filter-non-character.py --input $phrasecode.temp --output $phrasecode
echo "过滤完成，进行文本处理"
for split in train test valid
do
	echo "处理$split"
	inputfile=Data/$split.tok.en
	combinephrase=tempfile/temp.$split.phrasebpe2w.en
	python scripts/apply_bpe.py -c $phrasecode --input $inputfile --output $combinephrase --num-workers 1 --phrase-level True --sep " #$& "
	echo "文本短语bpe处理完成"

	wordcode=Data/bpe.code
	bothbpe=tempfile/temp.$split.phrasebpe.wordbpe.en
	echo "开始应用词级别bpe code处理文本"
	# 如果原始数据进行bpe的时候加了--vocabulary，那这里也得加。需要保持一致，否则recover的结果会不一样
	# subword-nmt apply-bpe -c $wordcode --vocabulary Data/vocab.en < $combinephrase > $bothbpe
	subword-nmt apply-bpe -c $wordcode < $combinephrase > $bothbpe
	echo "文本词bpe处理完成"

	spanfile=output/spans.$split.390w.phrase2wbpe.filter.en
	outputfile=output/final.$split.recover.en
	echo "开始抽取短语范围"
	python scripts/span-position.py --input $bothbpe --out $spanfile --withbpe
	echo "抽取短语范围完成，存入$spanfile"
	echo "开始清理文本中的占位符"
	python scripts/remove_placeholder.py --input $bothbpe --output $outputfile
	echo "清理完成，存入$outputfile，请检查是否与初始文件$inputfile一致"
	echo "请配套使用上述短语范围和清理后的文本"
	python scripts/filter-long-spans.py --input $spanfile --output $spanfile.clip10
	echo "过滤掉了长度大于10的span"

	if [ $ratio_status ]; then
		python scripts/count-span-ratio.py --originfile $inputfile --outputfile $spanfile.clip10
	fi
done


