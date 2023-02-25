# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 test_intent.py --test_file "${1}" --ckpt_path ./intent.pt?dl=0 --pred_file "${2}"