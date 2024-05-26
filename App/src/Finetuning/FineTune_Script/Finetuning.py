from App.src.Finetuning.HuggingFace import huggingface
from llama_index.finetuning import SentenceTransformersFinetuneEngine


class FineTune:
    def __init__(self, train_dataset, val_dataset, huggingfaceaccesstoken, model_id):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.huggingfaceaccesstoken = huggingfaceaccesstoken
        self.model_id = model_id

    def startfinetuning(self):
        hf = huggingface.HuggingFace()
        hf.huggingfacelogin(self.huggingfaceaccesstoken)
        model_id = self.model_id
        finetune_engine = SentenceTransformersFinetuneEngine(
            self.train_dataset,
            model_id=model_id,
            model_output_path=f"{self.folder_path}/finetuned_model",
            val_dataset=self.val_dataset,
            epochs=7,
            show_progress_bar=True,
        )
        finetune_engine.finetune()
