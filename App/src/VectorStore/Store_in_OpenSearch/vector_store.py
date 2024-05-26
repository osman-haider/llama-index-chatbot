from llama_index.vector_stores.opensearch import OpensearchVectorStore
from llama_index.core import VectorStoreIndex, StorageContext


class VectorStore:
    def __init__(self, client, nodes, embed_model):
        self.nodes = nodes
        self.client = client
        self.embed_model = embed_model
    def vec_store(self):
        # initialize vector store
        vector_store = OpensearchVectorStore(self.client)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # initialize an index using our sample data and the client we just created
        index = VectorStoreIndex(
            nodes=self.nodes,
            storage_context=storage_context,
            insert_batch_size=10,
            embed_model=self.embed_model,
            show_progress=True,
        )
        print("Vector Store is completed....")