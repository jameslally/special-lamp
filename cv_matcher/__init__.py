"""
CV Matcher Package

AI-powered CV-job matching using HuggingFace models with modular architecture.
"""

from .core.matcher import AICVMatcher
from .section_extraction.extractor import CVSectionExtractor
from .matching.analyzer import SectionAnalyzer
from .utils.file_handlers import FileHandler

__version__ = "1.0.0"
__author__ = "CV Matcher Team"

__all__ = [
    "AICVMatcher",
    "CVSectionExtractor", 
    "SectionAnalyzer",
    "FileHandler"
]
