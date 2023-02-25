train Context_selection
python Context_Selection.py --data_dir [path/to/your/data_dir]
train QA
python QA.py --data_dir [path/to/your/data_dir]
 
checkpoint will be saved at ./QA_{valid_acc}.pt and ./mul_{valid_acc}.pt
load model:
ckpt=torch.load(ckpt_path)
model.load_state_dict(ckpt['model_state_dict'])
