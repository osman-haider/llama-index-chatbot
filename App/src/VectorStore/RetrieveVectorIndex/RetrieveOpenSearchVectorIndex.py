from App.src.Finetuning.HuggingFace import huggingface
from App.src.VectorStore.Model import Embd_Model
from llama_index.vector_stores.opensearch import OpensearchVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

class RetrieveVector:
    def __init__(self, access_token):
        self.huggingface_instance = huggingface.HuggingFace()
        self.huggingface_instance.huggingfacelogin(access_token)

    def OpenSearchVectorIndexRetriver(self, model_id, client):
        load_model = Embd_Model.LoadModel(model_id)
        embed_model = load_model.loadmodel()
        # initialize vector store
        vector_store = OpensearchVectorStore(client)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # initialize an index using our sample data and the client we just created
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        return index


