from llama_index.embeddings.huggingface import HuggingFaceEmbedding
class LoadModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def loadmodel(self):
        print("Model is Ready for Download...")
        embed_model = HuggingFaceEmbedding(
            model_name=self.model_name
        )
        print("Downloading Complete")
        return embed_model