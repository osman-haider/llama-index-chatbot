from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine


def Query_Engine(index, custom_qa_prompt):
    llm = OpenAI(model="gpt-3.5-turbo-0613")

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=20,
    )

    response_synthesizer = get_response_synthesizer(text_qa_template=custom_qa_prompt, )

    default_query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        llm=llm,
    )

    return default_query_engine