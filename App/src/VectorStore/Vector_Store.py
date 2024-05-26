from App.src.Finetuning.HuggingFace import huggingface
from App.src.VectorStore.Model import Embd_Model
from App.src.Finetuning.Data_Preprocessing import Preprocessing
from App.src.VectorStore.Store_in_OpenSearch import vector_store
def vectorsearch(client, HUGGINGFACETOKEN, model_id, Folder_Path):
    hf = huggingface.HuggingFace
    hf.huggingfacelogin(HUGGINGFACETOKEN)
    embed_model = Embd_Model.LoadModel(model_id)
    preprocess_object = Preprocessing.DataPreprocess()
    nodes = preprocess_object.load_corpus(Folder_Path)
    vectorstore = vector_store.VectorStore(client, nodes, embed_model)
    vectorstore.vec_store()

