from App.src.Finetuning.Data_Preprocessing import Preprocessing
from App.src.Finetuning.FineTune_Script import Finetuning

def finetune(Folder_Path, OPENAPIKEY, HUGGINGFACETOKEN, model_id):
    # Initialize the fine-tuning object
    FineTuning = Preprocessing.DataPreprocess()
    FineTuning.setapikey(OPENAPIKEY)

    # Load corpus data and print verbose information
    nodes = FineTuning.load_corpus(Folder_Path, verbose=True)

    # Split the corpus nodes into training and validation sets
    train_nodes, val_nodes = FineTuning.splitting_nodes(nodes)

    # Create training and validation datasets from the nodes
    train_dataset, val_dataset = FineTuning.train_val_dataset_nodes(train_nodes, val_nodes,Folder_Path)

    # Finetuning
    FT = Finetuning.FineTune(train_dataset, val_dataset, HUGGINGFACETOKEN, model_id)
    FT.startfinetuning()

