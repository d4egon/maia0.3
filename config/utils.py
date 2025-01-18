# config/utils.py

from venv import logger
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSequenceClassification, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, pipeline
import torch
from keybert import KeyBERT # type: ignore

def get_sentence_transformer_model():
    """
    Returns an instance of the SentenceTransformer model used across the application.
    
    :return: SentenceTransformer model instance
    """
    return SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

def get_generation_model_and_tokenizer():
    """
    Returns the T5 model and tokenizer for response generation.
    
    :return: Tuple containing T5ForConditionalGeneration model and T5Tokenizer
    """
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    return model, tokenizer

def get_context_model_and_tokenizer():
    """
    Returns the BERT model and tokenizer for context understanding.
    
    :return: Tuple containing AutoModelForSequenceClassification model and AutoTokenizer
    """
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

def get_keyword_extractor(sentence_transformer_model):
    """
    Returns a KeyBERT extractor using the provided SentenceTransformer model.
    
    :param sentence_transformer_model: The SentenceTransformer model to use for keyword extraction.
    :return: KeyBERT extractor instance
    """
    return KeyBERT(model=sentence_transformer_model)

def get_visual_emotion_model():
    """
    Returns a pipeline for visual emotion analysis using ResNet-50.
    
    :return: Hugging Face pipeline for image classification
    """
    try:
        return pipeline("image-classification", model="microsoft/resnet-50")
    except Exception as e:
        logger.error(f"[INIT ERROR] Failed to load visual emotion model: {e}")
        raise

def get_whisper_model_and_pipeline():
    """
    Initialize and return the Whisper model and pipeline for speech-to-text conversion.

    :return: Tuple containing the Whisper model, processor, and pipeline
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"
    
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        pipeline_instance = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device
        )
        return model, processor, pipeline_instance
    except Exception as e:
        logger.error(f"[INIT ERROR] Failed to load Whisper model: {e}")
        raise
