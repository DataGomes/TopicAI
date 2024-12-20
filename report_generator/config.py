import os
from dotenv import load_dotenv
import pybliometrics.scopus
import voyageai
import google.generativeai as genai

def set_api_keys(pybliometrics_key: str = None, gemini_key: str = None, voyage_key: str = None):
    """
    Set API keys for the various services used by the TopicAI.
    If keys are not provided, it will attempt to load them from environment variables.
    
    Returns:
        bool: True if all keys are set successfully, False otherwise.
    """
    load_dotenv()  # Load environment variables from .env file if it exists

    if pybliometrics_key:
        os.environ['PYBLIOMETRICS_API_KEY'] = pybliometrics_key
    if voyage_key:
        os.environ['VOYAGE_API_KEY'] = voyage_key
    if gemini_key:
        os.environ['GEMINI_API_KEY'] = gemini_key

    try:
        # Initialize Scopus with the API key
        pybliometrics.scopus.init()

        # Set Voyage AI API key
        voyageai.api_key = os.getenv('VOYAGE_API_KEY')

        # Initialize Gemini client
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

        return True
    except Exception as e:
        print(f"Error setting API keys: {e}")
        return False
