mv ./QA.pt?dl=0 ./QA.pt
mv ./mul.pt?dl=0 ./mul.pt
mv ./config.json?dl=0 ./config.json
unzip tokenizer.zip?dl=0
python3 inference.py --context_path $1 --test_path $2 --output $3