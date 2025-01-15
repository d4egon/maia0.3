# self_initiated_conversation.py - Autonomous Conversations
import schedule
import time
import random
import threading
import logging
from typing import List, Dict

import socketio

from core import conversation_engine, memory_engine

import os
import tempfile
import pytest
from core.file_parser import FileParser
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
import docx
import mimetypes

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SelfInitiatedConversation:

    
    def __init__(self, memory_engine, conversation_engine, socketio):
        """
        Initialize the SelfInitiatedConversation with necessary components.

        :param memory_engine: An instance for managing memories.
        :param conversation_engine: An instance for processing conversation logic.
        :param socketio: Socket.IO server for real-time communication.
        """
        self.memory_engine = memory_engine
        self.conversation_engine = conversation_engine
        self.socketio = socketio
        
    def start_scheduler(self):
        """
        Start the scheduler for autonomous conversation triggers.
        """
        logger.info("[SCHEDULER START] Initiating autonomous conversation scheduler...")
        schedule.every().hour.do(self.trigger_reflection)
        schedule.every(30).minutes.do(self.trigger_conversation_start)
        threading.Thread(target=self.run_scheduler, daemon=True).start()

    def run_scheduler(self):
        """
        Run the scheduler in a loop to execute pending jobs.
        """
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"[SCHEDULER ERROR] {e}", exc_info=True)

    def trigger_reflection(self):
        """
        Trigger reflection by discussing top retrieved memories.
        """
        try:
            top_memories = self.memory_engine.get_top_retrieved_memories(limit=3)
            for memory in top_memories:
                response = self.conversation_engine.process_user_input(memory['text'])
                logger.info(f"[AUTONOMOUS REFLECTION] {response}")
                self.socketio.emit("new_message", {"message": response}, broadcast=True)
        except Exception as e:
            logger.error(f"[REFLECTION TRIGGER ERROR] {e}", exc_info=True)

    def trigger_conversation_start(self):
        """
        Initiate a new conversation with a random prompt.
        """
        prompts: List[str] = [
            "What have you been thinking about lately?",
            "Would you like to reflect on something together?",
            "Is there anything on your mind you'd like to explore?"
        ]
        try:
            chosen_prompt = random.choice(prompts)
            response = self.conversation_engine.process_user_input(chosen_prompt)
            logger.info(f"[AUTONOMOUS STARTER] {response}")
            self.socketio.emit("new_message", {"message": response}, broadcast=True)
        except Exception as e:
            logger.error(f"[CONVERSATION START ERROR] {e}", exc_info=True)

self_initiated_conversation = SelfInitiatedConversation(
        memory_engine, conversation_engine, socketio
    )

@pytest.fixture
def file_parser():
    return FileParser()

def create_temp_file(content, suffix):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if suffix == '.pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, content)
        pdf.output(temp_file.name)
    elif suffix == '.png':
        image = Image.new('RGB', (200, 100), color = (73, 109, 137))
        d = ImageDraw.Draw(image)
        d.text((10,10), content, fill=(255,255,0))
        image.save(temp_file.name)
    elif suffix == '.docx':
        doc = docx.Document()
        doc.add_paragraph(content)
        doc.save(temp_file.name)
    else:
        temp_file.write(content.encode('utf-8'))
    temp_file.close()
    return temp_file.name

def test_parse_pdf(file_parser):
    content = "This is a test PDF file."
    temp_file = create_temp_file(content, '.pdf')
    result = file_parser.parse(temp_file)
    assert content in result
    os.remove(temp_file)

def test_parse_image(file_parser):
    content = "This is a test image file."
    temp_file = create_temp_file(content, '.png')
    result = file_parser.parse(temp_file)
    assert "test image file" in result  # Use substring match
    os.remove(temp_file)

def test_parse_docx(file_parser):
    content = "This is a test DOCX file."
    temp_file = create_temp_file(content, '.docx')
    mime_type, _ = mimetypes.guess_type(temp_file)
    result = file_parser.parse(temp_file, mime_type)
    assert content in result
    os.remove(temp_file)

def test_parse_yaml(file_parser):
    content = "key: value"
    temp_file = create_temp_file(content, '.yaml')
    mime_type, _ = mimetypes.guess_type(temp_file)
    result = file_parser.parse(temp_file, mime_type)
    assert content in result
    os.remove(temp_file)

def test_parse_excel(file_parser):
    content = "This is a test Excel file."
    temp_file = create_temp_file(content, '.xlsx')
    mime_type, _ = mimetypes.guess_type(temp_file)
    result = file_parser.parse(temp_file, mime_type)
    assert content in result
    os.remove(temp_file)

def test_parse_unsupported(file_parser):
    content = "This is an unsupported file type."
    temp_file = create_temp_file(content, '.unsupported')
    mime_type, _ = mimetypes.guess_type(temp_file)
    with pytest.raises(ValueError, match="Unsupported file type"):
        file_parser.parse(temp_file, mime_type)
    os.remove(temp_file)