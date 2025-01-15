# Filename: /config/settings.py

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

CONFIG = {
    "NEO4J_URI": os.getenv("NEO4J_URI"),
    "NEO4J_USER": os.getenv("NEO4J_USER"),
    "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
}