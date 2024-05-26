from huggingface_hub import login
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

HUGGINGFACETOKEN = os.getenv('HUGGINGFACETOKEN')

# Create a token object
login(token=HUGGINGFACETOKEN)
