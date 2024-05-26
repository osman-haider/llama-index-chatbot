from llama_index.core import PromptTemplate

def Custom_prmpt():
    custom_qa_prompt = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n\n"
        "Note: If you don't get any context information, then just appologize that you don't "
        "have information about it rather than generating answer on your own.\n"
        "Furthermore, you are only required to communicate with user in ENGLISH LANGUAGE."
        "Query: {query_str}\n"
        "Answer: "
    )

    custom_qa_prompt = PromptTemplate(custom_qa_prompt)
    return custom_qa_prompt