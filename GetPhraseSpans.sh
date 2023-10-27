# hyperparameters
deltanum=100
operation_num=10000

# train set, after Tokenization but before BPE
inputtrainfile=Data/train.tok.en # [Fill in this position!] Write the file to learn WPE.
# where to save phrasecode
phrasecode=tempfile/ende14.delta$deltanum.1w.root.code

# wether to examine SU ratio after processing
ratio_status=true

echo "Learning WPE..."
python scripts/learn_wpe.py --input $inputtrainfile -s $operation_num --output $phrasecode.temp --delta $deltanum  --sep "#$&" --num-workers 1
echo "Finished the learning of WPE."
echo "Start to filter invalid semantic units/phrases..."
python scripts/filter-non-character.py --input $phrasecode.temp --output $phrasecode
echo "Finished filtering."
echo "Start to process texts..."
for split in train test valid
do
	echo "Deal with $split..."
	inputfile=Data/$split.tok.en # [Fill in this position!] Write the files that need to apply WPE.
	combinephrase=tempfile/temp.$split.phrasebpe1w.en
	echo "Start to apply WPE..."
	python scripts/apply_wpe.py -c $phrasecode --input $inputfile --output $combinephrase --num-workers 1 --phrase-level True --sep " #$& "
	echo "WPE applied to $split."

	# BPE code of subword-nmt
	wordcode=Data/bpe.code # [Fill in this position!] Write the pre-learned BPE code path.
	wordvocabulary=Data/vocab.en # [Fill in this position!] Write the pre-learned BPE vocabulary, if any.
	# where to save processed texts
	bothbpe=tempfile/temp.$split.phrasebpe.wordbpe.en
	echo "Start to apply original BPE..."
	# If "--vocabulary" is added when learning original BPE, you have to use it here to make sure the consistence.
	# Otherwise do not use it here. You can see the difference from the "recovered" file at last.
	# subword-nmt apply-bpe -c $wordcode --vocabulary $wordvocabulary < $combinephrase > $bothbpe
	subword-nmt apply-bpe -c $wordcode < $combinephrase > $bothbpe
	echo "Original BPE applied to $split."

	# where to save the boundaries of Semantic Units.
	spanfile=output/spans.$split.phrase1wbpe.filter.en
	# where to save the "recovered" file.
	outputfile=output/final.$split.recover.en
	echo "Start to extract semantic unit/phrase boundaries..."
	python scripts/span-position.py --input $bothbpe --out $spanfile --withbpe
	echo "Finished extraction of semantic unit/phrase boundaries, which are saved in $spanfile."
	echo "Start cleaning the separation signs (#$&)..."
	python scripts/remove_placeholder.py --input $bothbpe --output $outputfile
	echo "Finished cleaning, outputing to $outputfile."
	echo "Please check if $outputfile is the same with $inputfile, using 'ls -ll' for example."
	echo "NOTE: the SU boundary file have to match the output file $outputfile."
	python scripts/filter-long-spans.py --input $spanfile --threshold 10 --output $spanfile.clip10
	echo "Filtered spans with length longer than 10."

	if [ $ratio_status ]; then
		python scripts/count-span-ratio.py --originfile $inputfile --outputfile $spanfile.clip10
	fi
	echo "Finished $split set."
done


