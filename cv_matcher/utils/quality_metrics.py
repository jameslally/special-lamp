"""
Quality Metrics Utility

Analyzes the quality of CV section extraction and provides metrics.
"""

import re
from typing import Dict
from pathlib import Path


class QualityMetrics:
    """Analyzes the quality of CV section extraction."""
    
    @staticmethod
    def get_section_extraction_quality_metrics(cv_sections: Dict[str, str]) -> Dict:
        """Analyze the quality of AI-powered section extraction."""
        metrics = {}
        
        print("\n🔍 AI Section Extraction Quality Analysis:")
        print("=" * 50)
        
        total_content = sum(len(content) for content in cv_sections.values())
        
        for section_name, content in cv_sections.items():
            if content.strip():
                content_length = len(content)
                content_percentage = (content_length / total_content * 100) if total_content > 0 else 0
                
                # Analyze content quality
                sentences = len(re.split(r'[.!?]+', content))
                avg_sentence_length = content_length / sentences if sentences > 0 else 0
                
                # Determine quality indicators
                quality_score = QualityMetrics._calculate_quality_score(
                    content_length, sentences, avg_sentence_length, content
                )
                
                metrics[section_name] = {
                    'content_length': content_length,
                    'content_percentage': round(content_percentage, 2),
                    'sentences': sentences,
                    'avg_sentence_length': round(avg_sentence_length, 1),
                    'quality_score': quality_score,
                    'quality_level': QualityMetrics._get_quality_level(quality_score)
                }
                

        
        return metrics
    
    @staticmethod
    def _calculate_quality_score(content_length: int, sentences: int, avg_sentence_length: float, content: str) -> int:
        """Calculate quality score based on various metrics."""
        quality_score = 0
        
        # Content length scoring
        if content_length > 200:
            quality_score += 30  # Good length
        elif content_length > 100:
            quality_score += 20  # Moderate length
        elif content_length > 50:
            quality_score += 10  # Minimal length
        
        # Sentence count scoring
        if sentences >= 3:
            quality_score += 25  # Good sentence count
        elif sentences >= 2:
            quality_score += 15  # Moderate sentence count
        elif sentences >= 1:
            quality_score += 10  # Minimal sentence count
        
        # Sentence length scoring
        if avg_sentence_length > 50:
            quality_score += 25  # Good sentence length
        elif avg_sentence_length > 30:
            quality_score += 15  # Moderate sentence length
        elif avg_sentence_length > 15:
            quality_score += 10  # Minimal sentence length
        
        # Content relevance scoring
        if any(keyword in content.lower() for keyword in ["experience", "skills", "education", "summary"]):
            quality_score += 20  # Relevant content
        
        return quality_score
    
    @staticmethod
    def _get_quality_level(score: int) -> str:
        """Convert quality score to descriptive level."""
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Very Good"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Poor"
    
    @staticmethod
    def save_sections_to_file(cv_sections: Dict[str, str], output_file: str = "debug/extracted_cv_sections.txt") -> bool:
        """Save extracted CV sections to a readable text file for review."""
        try:
            # Create debug directory if it doesn't exist
            debug_dir = Path(output_file).parent
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("AI-POWERED CV SECTION EXTRACTION RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Total sections extracted: {len([s for s in cv_sections.values() if s.strip()])}\n")
                f.write(f"Total content length: {sum(len(s) for s in cv_sections.values())} characters\n\n")
                
                for section_name, content in cv_sections.items():
                    if content.strip():
                        f.write("-" * 60 + "\n")
                        f.write(f"SECTION: {section_name.upper()}\n")
                        f.write(f"Length: {len(content)} characters\n")
                        f.write(f"Sentences: {len(re.split(r'[.!?]+', content))}\n")
                        f.write("-" * 60 + "\n")
                        f.write(content)
                        f.write("\n\n")
                    else:
                        f.write("-" * 60 + "\n")
                        f.write(f"SECTION: {section_name.upper()}\n")
                        f.write("Status: No content detected\n")
                        f.write("-" * 60 + "\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("END OF SECTION EXTRACTION\n")
                f.write("=" * 80 + "\n")
            
            print(f"📁 CV sections saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"⚠ Failed to save sections to file: {e}")
            return False
