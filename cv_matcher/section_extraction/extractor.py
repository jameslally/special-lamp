"""
Optimized CV Section Extraction Module

Eliminates redundant AI calls and improves performance through smart caching and pattern-first approach.
"""

import re
import torch
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModel


class CVSectionExtractor:
    """Optimized CV section extractor with reduced AI calls."""
    
    def __init__(self, model_name: str = "allenai/longformer-base-4096"):
        """Initialize with shared model instance."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Initialize models (can be shared)
        self.tokenizer = None
        self.model = None
        
        # Cache for embeddings to avoid recomputation
        self.embedding_cache = {}
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load the AI model for semantic analysis."""
        try:
            print(f"🔧 Loading AI model for section extraction: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            print("✅ Section extraction model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load section extraction model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract CV sections using optimized approach."""

        
        # Initialize sections
        sections = {
            "summary": "",
            "experience": "",
            "skills": "",
            "education": "",
            "other": ""
        }
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # First pass: Identify section headers (FAST - no AI)
        section_boundaries = self._identify_section_boundaries_fast(paragraphs)
        
        # Second pass: Smart content classification
        current_section = "summary"  # Default starting section
        
        # Pre-classify paragraphs using fast pattern matching
        paragraph_classifications = self._fast_pattern_classification(paragraphs)
        
        for i, paragraph in enumerate(paragraphs):
            # Check if this paragraph is a section header
            if self._is_section_header(paragraph):
                current_section = self._get_section_from_header(paragraph)
                continue
                
            # Skip very short paragraphs
            if len(paragraph) < 20:
                continue
            
            # Use fast pattern classification first
            pattern_classification = paragraph_classifications[i]
            
            # Only use AI if pattern classification is uncertain
            if pattern_classification == "uncertain":
                # Use AI sparingly - only for ambiguous cases
                section_classification = self._ai_semantic_classification_optimized(paragraph)
                target_section = section_classification
            else:
                # Use pattern classification
                target_section = pattern_classification
                
                # If pattern suggests different section than current, respect CV structure
                if target_section != current_section and target_section != "other":
                    target_section = current_section
            
            # Add paragraph to the appropriate section
            if target_section in sections:
                sections[target_section] += paragraph + "\n\n"
            else:
                sections["other"] += paragraph + "\n\n"
        
        # Clean and validate sections
        cleaned_sections = {}
        for section_name, content in sections.items():
            if len(content.strip()) > 50:
                cleaned_sections[section_name] = content.strip()
            else:
                cleaned_sections[section_name] = ""
        

        
        return cleaned_sections
    
    def _identify_section_boundaries_fast(self, paragraphs: List[str]) -> Dict[str, List[int]]:
        """Fast section boundary identification using only regex patterns."""
        boundaries = {}
        
        # Common section header patterns
        section_patterns = {
            'summary': [r'\b(?:summary|profile|objective|overview|introduction)\b', r'\b(?:professional|executive)\s+(?:summary|profile)\b'],
            'experience': [r'\b(?:experience|work\s+history|employment|career|professional\s+background)\b', r'\b(?:work\s+experience|employment\s+history)\b'],
            'skills': [r'\b(?:skills|expertise|competencies|capabilities|technical\s+skills|core\s+skills)\b'],
            'education': [r'\b(?:education|academic|qualifications|degrees|certifications)\b']
        }
        
        for section_name, patterns in section_patterns.items():
            boundaries[section_name] = []
            for i, paragraph in enumerate(paragraphs):
                paragraph_lower = paragraph.lower()
                for pattern in patterns:
                    if re.search(pattern, paragraph_lower, re.IGNORECASE):
                        boundaries[section_name].append(i)
                        break
        
        return boundaries
    
    def _fast_pattern_classification(self, paragraphs: List[str]) -> List[str]:
        """Fast pattern-based classification without AI calls."""
        classifications = []
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            
            # Strong pattern indicators
            if self._is_summary_content(paragraph_lower):
                classifications.append("summary")
            elif self._is_experience_content(paragraph_lower):
                classifications.append("experience")
            elif self._is_skills_content(paragraph_lower):
                classifications.append("skills")
            elif self._is_education_content(paragraph_lower):
                classifications.append("education")
            else:
                # Mark as uncertain for potential AI analysis
                classifications.append("uncertain")
        
        return classifications
    
    def _is_section_header(self, paragraph: str) -> bool:
        """Check if a paragraph is likely a section header."""
        if len(paragraph) < 100:
            header_keywords = [
                'summary', 'profile', 'objective', 'overview', 'introduction',
                'experience', 'work history', 'employment', 'career',
                'skills', 'expertise', 'competencies', 'capabilities',
                'education', 'academic', 'qualifications', 'degrees', 'certifications'
            ]
            
            paragraph_lower = paragraph.lower()
            return any(keyword in paragraph_lower for keyword in header_keywords)
        
        return False
    
    def _get_section_from_header(self, header_text: str) -> str:
        """Determine section type from header text."""
        header_lower = header_text.lower().strip()
        
        if any(keyword in header_lower for keyword in ["summary", "profile", "objective", "overview", "introduction"]):
            return "summary"
        elif any(keyword in header_lower for keyword in ["experience", "work history", "employment", "career", "professional background"]):
            return "experience"
        elif any(keyword in header_lower for keyword in ["skills", "expertise", "competencies", "capabilities", "core skills", "technical skills"]):
            return "skills"
        elif any(keyword in header_lower for keyword in ["education", "academic", "qualifications", "degrees", "certifications"]):
            return "education"
        else:
            return "other"
    
    def _is_summary_content(self, text: str) -> bool:
        """Check if text contains summary/profile content."""
        summary_patterns = [
            r'\b(?:i\s+am|i\'m)\s+(?:a|an)\s+',  # "I am a..." or "I'm a..."
            r'\b(?:passionate|dedicated|experienced|skilled)\s+',  # Personal qualities
            r'\b(?:professional|leader|specialist|expert)\s+',     # Professional titles
            r'\b(?:enabling|driving|leading|transforming)\s+',     # Action-oriented statements
            r'\b(?:over\s+\d+\s+years?\s+of\s+experience)\b',    # Experience statements
            r'\b(?:specializing\s+in|focusing\s+on|expertise\s+in)\b'  # Specialization
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in summary_patterns)
    
    def _is_experience_content(self, text: str) -> bool:
        """Check if text contains work experience content."""
        experience_patterns = [
            r'\b(?:led|managed|developed|implemented|delivered)\s+',  # Action verbs
            r'\b(?:responsible\s+for|oversaw|supervised|coordinated)\b',  # Management
            r'\b(?:team\s+of\s+\d+|reported\s+to|managed\s+\d+)\b',     # Team size
            r'\b(?:project|initiative|program|strategy)\b',              # Project work
            r'\b(?:budget|cost|revenue|efficiency|improvement)\b',       # Business impact
            r'\b(?:client|customer|stakeholder|vendor|partner)\b',       # External relations
            r'\b(?:agile|scrum|waterfall|methodology|framework)\b'      # Process
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in experience_patterns)
    
    def _is_skills_content(self, text: str) -> bool:
        """Check if text contains technical skills content."""
        skills_patterns = [
            r'\b(?:proficient|expert|skilled|experienced|knowledge)\s+in\b',  # Skill level
            r'\b(?:technologies?|tools?|platforms?|frameworks?|languages?)\b', # Tech terms
            r'\b(?:aws|azure|gcp|docker|kubernetes|terraform|ansible)\b',     # Cloud/DevOps
            r'\b(?:python|java|javascript|react|angular|vue|node\.js)\b',     # Programming
            r'\b(?:mysql|postgresql|mongodb|redis|elasticsearch)\b',          # Databases
            r'\b(?:git|github|jenkins|agile|scrum|devops|ci/cd)\b',          # Tools/Processes
            r'\b(?:machine\s+learning|ai|data\s+science|cybersecurity)\b'    # Specialized
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in skills_patterns)
    
    def _is_education_content(self, text: str) -> bool:
        """Check if text contains education content."""
        education_patterns = [
            r'\b(?:university|college|institute|school|academy)\b',  # Institutions
            r'\b(?:bachelor|master|phd|doctorate|degree|diploma)\b', # Degrees
            r'\b(?:certification|certified|license|accreditation)\b', # Certifications
            r'\b(?:graduated|completed|studied|majored|minored)\b',   # Academic actions
            r'\b(?:gpa|grade|honors|dean\'s\s+list|valedictorian)\b', # Academic performance
            r'\b(?:thesis|dissertation|research|publication)\b'       # Academic work
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in education_patterns)
    
    def _ai_semantic_classification_optimized(self, paragraph: str) -> str:
        """Optimized AI semantic classification with caching."""
        # Check cache first
        if paragraph in self.embedding_cache:
            return self.embedding_cache[paragraph]
        
        try:
            # Only use AI for truly ambiguous cases
            # Create semantic embeddings for the paragraph
            paragraph_embedding = self._get_semantic_embedding(paragraph)
            
            # Define section templates for semantic comparison
            section_templates = {
                "summary": ["professional summary", "career objective", "personal profile"],
                "experience": ["work experience", "employment history", "professional experience"],
                "skills": ["technical skills", "core competencies", "expertise areas"],
                "education": ["academic background", "educational qualifications", "degrees"]
            }
            
            best_match = "other"
            highest_similarity = 0.0
            
            # Compare paragraph with each section template
            for section_name, templates in section_templates.items():
                for template in templates:
                    template_embedding = self._get_semantic_embedding(template)
                    similarity = self._cosine_similarity(paragraph_embedding, template_embedding)
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = section_name
            
            # Cache the result
            self.embedding_cache[paragraph] = best_match
            return best_match
            
        except Exception as e:
            print(f"⚠️ AI classification failed, using fallback: {e}")
            return "other"
    
    def _get_semantic_embedding(self, text: str) -> torch.Tensor:
        """Get semantic embedding for text using the AI model."""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.squeeze(0)
                
        except Exception as e:
            print(f"⚠️ Failed to get semantic embedding: {e}")
            return torch.zeros(768)
    
    def _cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1_norm = vec1 / (torch.norm(vec1) + 1e-8)
            vec2_norm = vec2 / (torch.norm(vec2) + 1e-8)
            similarity = torch.dot(vec1_norm, vec2_norm).item()
            return max(0, similarity)
            
        except Exception as e:
            print(f"⚠️ Cosine similarity calculation failed: {e}")
            return 0.0
    
    def get_model_instance(self):
        """Get the model instance for sharing with other components."""
        return self.model, self.tokenizer
    
    def clear_cache(self):
        """Clear the embedding cache to free memory."""
        self.embedding_cache.clear()
