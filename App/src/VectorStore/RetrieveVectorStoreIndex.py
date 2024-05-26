from App.src.VectorStore.RetrieveVectorIndex import RetrieveOpenSearchVectorIndex

def Retrive_Vector_Store(client, HUGGINGFACETOKEN, model_id):
    Retrierver = RetrieveOpenSearchVectorIndex.RetrieveVector(HUGGINGFACETOKEN)
    index = Retrierver.OpenSearchVectorIndexRetriver(model_id, client)
    if index:
        print("Index is Loaded...")
        return index
    else:
        print("Index is not Loaded...")