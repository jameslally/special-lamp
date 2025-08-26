"""
Optimized CV Matcher Class

Shares AI models between components to eliminate redundancy and improve performance.
"""

import json
from pathlib import Path
from typing import Dict

from ..section_extraction.extractor import CVSectionExtractor
from ..matching.analyzer import SectionAnalyzer
from ..utils.file_handlers import FileHandler
from ..utils.quality_metrics import QualityMetrics


class AICVMatcher:
    """Optimized AI-powered CV matching with shared models."""
    
    def __init__(self, model_name: str = "allenai/longformer-base-4096"):
        """Initialize with shared model instances."""
        print(f"🤖 Initializing OPTIMIZED AI CV Matcher with model: {model_name}")
        
        # Initialize section extractor first (it loads the model)
        self.section_extractor = CVSectionExtractor(model_name)
        
        # Get shared model instances
        shared_model, shared_tokenizer = self.section_extractor.get_model_instance()
        
        # Initialize analyzer with shared models (no reloading)
        self.section_analyzer = SectionAnalyzer(model_name, shared_model, shared_tokenizer)
        
        # Initialize file handler
        self.file_handler = FileHandler()
        
        print("✅ OPTIMIZED AI CV Matcher initialized successfully")
        print("💡 Models shared between components - no redundant loading!")
    
    def comprehensive_analysis(self, cv_file: str, jd_file: str) -> Dict:
        """Perform comprehensive CV-job matching analysis with optimizations."""
        print(f"📄 Analyzing CV: {cv_file}")
        print(f"💼 Analyzing Job Description: {jd_file}")
        print("=" * 60)
        
        # Read files
        cv_text = self._read_cv_file(cv_file)
        jd_text = self._read_jd_file(jd_file)
        
        if not cv_text or not jd_text:
            print("❌ Failed to read input files")
            return {}
        
        # Clean texts
        cv_text = self.file_handler.clean_text(cv_text)
        jd_text = self.file_handler.clean_text(jd_text)
        
        # STEP 1: Extract CV sections using AI
        print("🔍 Extracting CV sections...")
        cv_sections = self.section_extractor.extract_sections(cv_text)
        
        # Save sections to file for review
        QualityMetrics.save_sections_to_file(cv_sections)
        
        # STEP 2: Analyze each section individually with the AI model
        print("🧠 Analyzing sections with AI...")
        section_scores = self.section_analyzer.analyze_sections(cv_sections, jd_text)
        
        # STEP 3: Calculate overall score based on section scores
        print("📊 Calculating overall score...")
        overall_score = self.section_analyzer.calculate_overall_score(section_scores, cv_sections)
        
        # Get section extraction quality metrics
        section_quality_metrics = QualityMetrics.get_section_extraction_quality_metrics(cv_sections)
        

        
        # Clear cache to free memory
        self.section_extractor.clear_cache()
        
        # Compile results
        results = {
            'overall_score': overall_score,
            'section_scores': section_scores,
            'section_insights': self.section_analyzer.get_section_insights(),
            'section_quality_metrics': section_quality_metrics,
            'cv_sections': cv_sections,
            'cv_text_length': len(cv_text),
            'jd_text_length': len(jd_text)
        }
        
        return results
    
    def _read_cv_file(self, cv_file: str) -> str:
        """Read CV file based on its extension."""
        if self.file_handler.get_file_extension(cv_file) == '.docx':
            return self.file_handler.read_docx(cv_file)
        else:
            return self.file_handler.read_txt(cv_file)
    
    def _read_jd_file(self, jd_file: str) -> str:
        """Read job description file."""
        return self.file_handler.read_txt(jd_file)
    
    def save_results(self, results: Dict, output_file: str = "debug/ai_matching_results.json"):
        """Save analysis results to JSON file."""
        try:
            # Create debug directory if it doesn't exist
            debug_dir = Path(output_file).parent
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"⚠ Failed to save results: {e}")
    
    def get_section_extractor(self) -> CVSectionExtractor:
        """Get the optimized section extractor component."""
        return self.section_extractor
    
    def get_section_analyzer(self) -> SectionAnalyzer:
        """Get the section analyzer component."""
        return self.section_analyzer
    
    def get_file_handler(self) -> FileHandler:
        """Get the file handler component."""
        return self.file_handler
