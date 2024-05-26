import random
import openai
from llama_index.llms.openai import OpenAI

from App.src.Finetuning.Prompt.prompt import QA_GENERATE_PROMPT_TMPL

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset


class DataPreprocess:
    def setapikey(self, openai_api_key):
        openai.api_key(openai_api_key)

    def load_corpus(self, folder_path, verbose=False):
        if verbose:
            print(f"Loading files from the path: {folder_path}")

        reader = SimpleDirectoryReader(folder_path)
        docs = reader.load_data()
        if verbose:
            print(f"Loaded {len(docs)} docs")

        splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=120,
        )

        nodes = splitter.get_nodes_from_documents(docs, show_progress=verbose)
        if verbose:
            print(f"Parsed {len(nodes)} nodes")

        return nodes

    def splitting_nodes(self, nodes, split_ratio=0.8):
        random.shuffle(nodes)
        split_ratio = split_ratio
        split_index = int(split_ratio * len(nodes))

        train_nodes = nodes[:split_index]
        val_nodes = nodes[split_index:]
        return train_nodes, val_nodes

    def train_val_dataset_nodes(self, train_nodes, val_nodes, folder_path):

        train_dataset = generate_qa_embedding_pairs(
            nodes=train_nodes,
            llm=OpenAI(model="gpt-3.5-turbo"),
            qa_generate_prompt_tmpl=QA_GENERATE_PROMPT_TMPL,
            num_questions_per_chunk=5,
        )

        val_dataset = generate_qa_embedding_pairs(
            nodes=val_nodes,
            llm=OpenAI(model="gpt-3.5-turbo"),
            qa_generate_prompt_tmpl=QA_GENERATE_PROMPT_TMPL,
            num_questions_per_chunk=5,
        )
        train_dataset.save_json(f"{folder_path}/train_dataset.json")
        val_dataset.save_json(f"{folder_path}/val_dataset.json")
        print("Train and Val Dataset is created")
        return train_dataset, val_dataset
    def load_dataset(self, tarin_dataset_file_path, test_dataset_file_path):
        # Load
        train_dataset = EmbeddingQAFinetuneDataset.from_json(tarin_dataset_file_path)
        val_dataset = EmbeddingQAFinetuneDataset.from_json(test_dataset_file_path)
