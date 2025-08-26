"""
File handling utilities for CV and job description files.
"""

import docx
import re
from pathlib import Path
from typing import Optional


class FileHandler:
    """Handles file reading and text cleaning operations."""
    
    @staticmethod
    def read_docx(file_path: str) -> str:
        """Read text from a .docx CV file."""
        try:
            doc = docx.Document(file_path)
            # Preserve paragraph structure by joining with newlines
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text
        except Exception as e:
            print(f"⚠ Error reading DOCX file: {e}")
            return ""
    
    @staticmethod
    def read_txt(file_path: str) -> str:
        """Read text from a .txt file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠ Error reading TXT file: {e}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text while preserving paragraph breaks."""
        # Remove extra whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Only normalize spaces and tabs, not newlines
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\-\+\&\@\#\n]', '', text)  # Keep newlines
        return text.strip()
    
    @staticmethod
    def save_text_to_file(content: str, output_file: str, encoding: str = 'utf-8') -> bool:
        """Save text content to a file."""
        try:
            with open(output_file, 'w', encoding=encoding) as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"⚠ Failed to save file: {e}")
            return False
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """Get file extension from file path."""
        return Path(file_path).suffix.lower()
    
    @staticmethod
    def file_exists(file_path: str) -> bool:
        """Check if file exists."""
        return Path(file_path).exists()
