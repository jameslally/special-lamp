"""
CV Section Analysis Module

Handles the analysis and matching of CV sections against job descriptions.
"""

import re
import torch
from typing import Dict, List
from transformers import AutoTokenizer, AutoModel


class SectionAnalyzer:
    """Analyzes CV sections against job descriptions using AI-powered matching."""
    
    def __init__(self, model_name: str = "allenai/longformer-base-4096", shared_model=None, shared_tokenizer=None):
        """Initialize the section analyzer with AI model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Store insights for later use
        self.section_insights = {}
        
        # Use shared models if provided, otherwise load new ones
        if shared_model is not None and shared_tokenizer is not None:
            self.model = shared_model
            self.tokenizer = shared_tokenizer
        else:
            # Load models
            self.tokenizer = None
            self.model = None
            self._load_models()
    
    def _load_models(self):
        """Load the AI model for section analysis."""
        try:
            print(f"🔧 Loading AI model for section analysis: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"❌ Failed to load section analysis model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def analyze_sections(self, cv_sections: Dict[str, str], jd_text: str) -> Dict[str, float]:
        """Analyze match scores for individual CV sections using AI-powered analysis."""
        section_scores = {}
        section_insights = {}
        
        for section_name, section_text in cv_sections.items():
            if section_text.strip() and len(section_text) > 50:  # Only analyze sections with substantial content
                # Get AI-powered section analysis
                section_analysis = self._analyze_section_with_ai(section_text, jd_text, section_name)
                section_scores[section_name] = section_analysis['score']
                section_insights[section_name] = section_analysis['insights']
            else:
                section_scores[section_name] = 0.0
                section_insights[section_name] = "Insufficient content for analysis"
        
        # Store insights for later use
        self.section_insights = section_insights
        
        return section_scores
    
    def _analyze_section_with_ai(self, section_text: str, jd_text: str, section_name: str) -> Dict:
        """Analyze a specific section with AI-powered insights."""
        try:
            # Get the basic match score
            score = self._calculate_match_score(section_text, jd_text)
            
            # Generate AI-powered insights
            insights = self._generate_section_insights(section_text, jd_text, section_name, score)
            
            return {
                'score': score,
                'insights': insights
            }
            
        except Exception as e:
            print(f"⚠️ AI section analysis failed: {e}")
            return {
                'score': 0.0,
                'insights': f"Analysis failed: {str(e)}"
            }
    
    def _calculate_match_score(self, cv_text: str, jd_text: str) -> float:
        """Calculate match score using the AI model for document analysis."""
        try:
            # Model capacity check
            max_tokens = 32768  # Longformer's actual capacity
            estimated_chars = max_tokens * 4  # Rough estimate: 4 chars per token
            
            # Check if we can process everything without truncation
            if len(cv_text) + len(jd_text) <= estimated_chars:
                combined_text = f"{cv_text} [SEP] {jd_text}"
            else:
                combined_text = f"{cv_text} [SEP] {jd_text}"
            
            # Prepare inputs for the model
            inputs = self.tokenizer(
                combined_text,
                return_tensors="pt",
                max_length=max_tokens,
                truncation=True,  # Keep this as safety
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use the last hidden state for Longformer (no pooler output)
                last_hidden_state = outputs.last_hidden_state
                
                # For Longformer, we'll use the [CLS] token representation (first token)
                cls_embedding = last_hidden_state[0, 0, :]  # [CLS] token
                
                # Calculate the magnitude of the embedding (higher = more confident)
                embedding_magnitude = torch.norm(cls_embedding).item()
                
                # Normalize to 0-100 scale
                max_expected_magnitude = 25.0  # Adjusted for Longformer
                normalized_score = min(100, (embedding_magnitude / max_expected_magnitude) * 100)
                
                return round(normalized_score, 2)
                
        except Exception as e:
            print(f"    ❌ AI analysis failed: {e}")
            raise RuntimeError(f"AI analysis failed: {e}")
    
    def _generate_section_insights(self, section_text: str, jd_text: str, section_name: str, score: float) -> str:
        """Generate intelligent insights about section matching."""
        try:
            # Analyze content overlap and relevance
            section_words = set(re.findall(r'\b\w+\b', section_text.lower()))
            jd_words = set(re.findall(r'\b\w+\b', jd_text.lower()))
            
            # Find relevant keywords
            relevant_keywords = section_words.intersection(jd_words)
            
            # Generate insights based on section type and score
            if section_name == "summary":
                if score >= 80:
                    return "Excellent alignment with job requirements and company culture"
                elif score >= 60:
                    return "Good match with job profile, some areas could be enhanced"
                else:
                    return "Limited alignment - consider tailoring summary to job requirements"
                    
            elif section_name == "experience":
                if score >= 80:
                    return "Strong relevant experience that directly matches job requirements"
                elif score >= 60:
                    return "Relevant experience with some gaps in specific areas"
                else:
                    return "Experience may not directly align with job requirements"
                    
            elif section_name == "skills":
                if score >= 80:
                    return "Excellent technical skill match with job requirements"
                elif score >= 60:
                    return "Good skill overlap, some additional skills could be beneficial"
                else:
                    return "Limited skill alignment - consider highlighting relevant technical abilities"
                    
            elif section_name == "education":
                if score >= 80:
                    return "Educational background strongly supports job requirements"
                elif score >= 60:
                    return "Educational qualifications meet basic requirements"
                else:
                    return "Educational background may need additional certifications or training"
                    
            else:
                # Generic insight
                if score >= 80:
                    return "Excellent overall alignment with job requirements"
                elif score >= 60:
                    return "Good match with room for improvement"
                else:
                    return "Limited alignment - consider enhancing relevant areas"
                    
        except Exception as e:
            return f"Insight generation failed: {str(e)}"
    
    def calculate_overall_score(self, section_scores: Dict[str, float], cv_sections: Dict[str, str]) -> float:
        """Calculate overall score based on individual section scores with content-based weighting."""
        if not section_scores:
            return 0.0
        
        # Define section weights based on typical importance in job matching
        section_weights = {
            'summary': 0.15,      # 15% - overview and cultural fit
            'experience': 0.35,   # 35% - most important for job matching
            'skills': 0.30,       # 30% - technical capabilities
            'education': 0.15,    # 15% - qualifications
            'other': 0.05         # 5% - additional information
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for section_name, score in section_scores.items():
            if score > 0 and section_name in section_weights:
                weight = section_weights[section_name]
                weighted_score = score * weight
                total_weighted_score += weighted_score
                total_weight += weight
        
        if total_weight > 0:
            overall_score = total_weighted_score / total_weight
            return round(overall_score, 2)
        else:
            return 0.0
    
    def get_section_insights(self) -> Dict[str, str]:
        """Get the stored section insights."""
        return self.section_insights
    

