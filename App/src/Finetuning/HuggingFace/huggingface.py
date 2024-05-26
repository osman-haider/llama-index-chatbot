from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

class HuggingFace:
    def huggingfacelogin(self, access_token):
        login(token=access_token)

    def push_model(self, folder_path, huggingface_model_path):
        # Load the fine-tuned model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(folder_path)
        tokenizer = AutoTokenizer.from_pretrained(folder_path)

        # Push the model to the Hugging Face model hub
        model.push_to_hub(huggingface_model_path, use_auth_token=True)
