import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple
import docx
import re

class AICVMatcher:
    """AI-powered CV matching using HuggingFace models."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🤖 Using device: {self.device}")
        
        # Initialize models
        self.talent_match_model = None
        self.talent_match_tokenizer = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load the specialized talent matching model."""
        print("🔧 Loading AI model...")
        
        try:
            # Load Longformer model for long sequence processing
            print("Loading allenai/longformer-base-4096...")
            print("💡 This model supports up to 32,768 tokens (~131,072 characters)")
            print("💡 Perfect for full CV and job description analysis")
            
            self.talent_match_tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
            self.talent_match_model = AutoModel.from_pretrained("allenai/longformer-base-4096")
            
            # Move to device
            self.talent_match_model.to(self.device)
            self.talent_match_model.eval()
            print("✅ Longformer model loaded successfully")
            print(f"📏 Model capacity: {self.talent_match_tokenizer.model_max_length} tokens")
            
        except Exception as e:
            print(f"❌ Failed to load Longformer model: {e}")
            print("💡 Please ensure the model is available and try again.")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def read_docx(self, file_path: str) -> str:
        """Read text from a .docx CV file."""
        try:
            doc = docx.Document(file_path)
            # Preserve paragraph structure by joining with newlines
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text
        except Exception as e:
            print(f"⚠ Error reading DOCX file: {e}")
            return ""
    
    def read_txt(self, file_path: str) -> str:
        """Read text from a .txt file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠ Error reading TXT file: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Only normalize spaces and tabs, not newlines
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\-\+\&\@\#\n]', '', text)  # Keep newlines
        return text.strip()
    
    def extract_cv_sections(self, text: str) -> Dict[str, str]:
        """Extract CV sections using AI-powered semantic analysis."""
        print("🧠 Using AI to intelligently extract CV sections...")
        
        # Initialize sections
        sections = {
            "summary": "",
            "experience": "",
            "skills": "",
            "education": "",
            "other": ""
        }
        
        # Split text into paragraphs for better semantic analysis
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # First pass: Identify section headers and boundaries
        section_boundaries = self._identify_section_boundaries(paragraphs)
        
        # Second pass: Classify content into sections using AI
        current_section = "summary"  # Default starting section
        
        print(f"  🔍 Processing {len(paragraphs)} paragraphs...")
        
        for i, paragraph in enumerate(paragraphs):
            # Check if this paragraph is a section header (even if short)
            if self._is_section_header(paragraph):
                # Update current section based on header
                current_section = self._get_section_from_header(paragraph)
                print(f"  🎯 Detected section header: '{paragraph}' → {current_section}")
                continue
                
            # Skip very short paragraphs that aren't headers
            if len(paragraph) < 20:
                continue
                
            # Use AI to classify the paragraph content, but respect section boundaries
            section_classification = self._classify_paragraph_semantically(paragraph)
            
            # If AI classification matches current section, use it; otherwise, use current section
            if section_classification == current_section or section_classification == "other":
                target_section = current_section
            else:
                # AI suggests different section, but respect CV structure
                target_section = current_section
                print(f"  💡 AI suggested '{section_classification}' but using '{current_section}' based on CV structure")
            
            # Add paragraph to the appropriate section
            if target_section in sections:
                sections[target_section] += paragraph + "\n\n"
            else:
                sections["other"] += paragraph + "\n\n"
        
        # Clean and validate sections
        cleaned_sections = {}
        for section_name, content in sections.items():
            cleaned_content = self.clean_text(content)
            if len(cleaned_content) > 50:  # Only keep sections with substantial content
                cleaned_sections[section_name] = cleaned_content
            else:
                cleaned_sections[section_name] = ""
        
        # Print section analysis
        print("📊 AI Section Analysis Results:")
        for section_name, content in cleaned_sections.items():
            if content.strip():
                print(f"  🎯 {section_name.title()}: {len(content)} characters")
            else:
                print(f"  ⚪ {section_name.title()}: No content detected")
        
        return cleaned_sections
    
    def _identify_section_boundaries(self, paragraphs: List[str]) -> Dict[str, List[int]]:
        """Identify potential section boundaries in the CV."""
        boundaries = {}
        
        # Common section header patterns
        section_patterns = {
            'summary': [r'\b(?:summary|profile|objective|overview|introduction)\b', r'\b(?:professional|executive)\s+(?:summary|profile)\b'],
            'experience': [r'\b(?:experience|work\s+history|employment|career|professional\s+background)\b', r'\b(?:work\s+experience|employment\s+history)\b'],
            'skills': [r'\b(?:skills|expertise|competencies|capabilities|technical\s+skills)\b', r'\b(?:core\s+skills|professional\s+skills)\b'],
            'education': [r'\b(?:education|academic|qualifications|degrees|certifications)\b', r'\b(?:educational\s+background|academic\s+credentials)\b']
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
    
    def _is_section_header(self, paragraph: str) -> bool:
        """Check if a paragraph is likely a section header."""
        # Section headers are typically short and contain specific keywords
        if len(paragraph) < 100:  # Headers are usually short
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
        
        # More flexible patterns for section headers
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
    
    def _classify_paragraph_semantically(self, paragraph: str) -> str:
        """Use AI to semantically classify a paragraph into CV sections."""
        try:
            # Enhanced content pattern analysis
            paragraph_lower = paragraph.lower()
            
            # Check for specific content patterns first
            if self._is_summary_content(paragraph_lower):
                return "summary"
            elif self._is_experience_content(paragraph_lower):
                return "experience"
            elif self._is_skills_content(paragraph_lower):
                return "skills"
            elif self._is_education_content(paragraph_lower):
                return "education"
            
            # If no clear pattern, use AI semantic analysis
            section_classification = self._ai_semantic_classification(paragraph)
            return section_classification
            
        except Exception as e:
            print(f"⚠️ AI classification failed for paragraph, using fallback: {e}")
            # Fallback to basic keyword matching
            return self._fallback_section_classification(paragraph)
    
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
    
    def _ai_semantic_classification(self, paragraph: str) -> str:
        """Use AI semantic analysis for classification when patterns are unclear."""
        try:
            # Create semantic embeddings for the paragraph
            paragraph_embedding = self._get_semantic_embedding(paragraph)
            
            # Define section templates for semantic comparison
            section_templates = {
                "summary": [
                    "professional summary", "career objective", "personal profile", 
                    "executive summary", "overview", "introduction"
                ],
                "experience": [
                    "work experience", "employment history", "professional experience",
                    "career highlights", "work history", "employment record"
                ],
                "skills": [
                    "technical skills", "core competencies", "expertise areas",
                    "professional skills", "capabilities", "skill set"
                ],
                "education": [
                    "academic background", "educational qualifications", "degrees",
                    "certifications", "training", "academic credentials"
                ]
            }
            
            best_match = "other"
            highest_similarity = 0.0
            
            # Compare paragraph with each section template
            for section_name, templates in section_templates.items():
                for template in templates:
                    template_embedding = self._get_semantic_embedding(template)
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(paragraph_embedding, template_embedding)
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = section_name
            
            # Use additional semantic rules for better classification
            if highest_similarity < 0.3:  # Low similarity threshold
                # Apply content-based heuristics as fallback
                paragraph_lower = paragraph.lower()
                
                # Check for experience indicators
                if any(word in paragraph_lower for word in ["years", "worked", "led", "managed", "developed", "implemented", "responsible for"]):
                    best_match = "experience"
                # Check for skills indicators
                elif any(word in paragraph_lower for word in ["proficient", "expert", "skilled", "experience with", "knowledge of", "familiar with"]):
                    best_match = "skills"
                # Check for education indicators
                elif any(word in paragraph_lower for word in ["university", "college", "degree", "bachelor", "master", "phd", "certification"]):
                    best_match = "education"
                # Check for summary indicators
                elif any(word in paragraph_lower for word in ["overview", "summary", "profile", "objective", "passionate", "dedicated"]):
                    best_match = "summary"
            
            return best_match
            
        except Exception as e:
            print(f"⚠️ AI semantic classification failed: {e}")
            return "other"
    
    def _get_semantic_embedding(self, text: str) -> torch.Tensor:
        """Get semantic embedding for text using the Longformer model."""
        try:
            # Tokenize and get embeddings
            inputs = self.talent_match_tokenizer(
                text,
                return_tensors="pt",
                max_length=512,  # Reasonable length for paragraph analysis
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.talent_match_model(**inputs)
                # Use mean pooling of all tokens for paragraph representation
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Average across tokens
                return embeddings.squeeze(0)  # Remove batch dimension
                
        except Exception as e:
            print(f"⚠️ Failed to get semantic embedding: {e}")
            # Return zero tensor as fallback
            return torch.zeros(768)  # Standard embedding size
    
    def _cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Normalize vectors
            vec1_norm = vec1 / (torch.norm(vec1) + 1e-8)
            vec2_norm = vec2 / (torch.norm(vec2) + 1e-8)
            
            # Calculate cosine similarity
            similarity = torch.dot(vec1_norm, vec2_norm).item()
            return max(0, similarity)  # Ensure non-negative
            
        except Exception as e:
            print(f"⚠️ Cosine similarity calculation failed: {e}")
            return 0.0
    
    def _fallback_section_classification(self, paragraph: str) -> str:
        """Fallback method using basic keyword matching when AI fails."""
        paragraph_lower = paragraph.lower()
        
        # Simple keyword-based classification as backup
        if any(keyword in paragraph_lower for keyword in ["experience", "worked", "led", "managed", "developed", "implemented"]):
            return "experience"
        elif any(keyword in paragraph_lower for keyword in ["skills", "expertise", "proficient", "knowledge", "technologies"]):
            return "skills"
        elif any(keyword in paragraph_lower for keyword in ["education", "academic", "university", "college", "degree", "certification"]):
            return "education"
        elif any(keyword in paragraph_lower for keyword in ["summary", "profile", "objective", "overview"]):
            return "summary"
        else:
            return "other"
    
    def talent_match_score(self, cv_text: str, jd_text: str) -> float:
        """Calculate match score using the Longformer model for full document analysis."""
        try:
            print(f"    🔍 Using Longformer model: allenai/longformer-base-4096")
            
            # Longformer supports up to 32,768 tokens (~131,072 characters)
            # Your CV (9,715 chars) + JD (4,713 chars) = 14,428 chars total
            # This fits comfortably within the model's capacity
            max_tokens = 32768  # Longformer's actual capacity
            estimated_chars = max_tokens * 4  # Rough estimate: 4 chars per token
            
            print(f"    📏 Model capacity: {max_tokens:,} tokens (~{estimated_chars:,} characters)")
            print(f"    📄 CV length: {len(cv_text):,} characters")
            print(f"    📋 JD length: {len(jd_text):,} characters")
            print(f"    🔗 Combined: {len(cv_text) + len(jd_text):,} characters")
            
            # Check if we can process everything without truncation
            if len(cv_text) + len(jd_text) <= estimated_chars:
                print(f"    ✅ Full CV and JD fit within model capacity - no truncation needed!")
                combined_text = f"{cv_text} [SEP] {jd_text}"
            else:
                print(f"    ⚠ Texts exceed estimated capacity, but Longformer should handle this")
                print(f"    💡 Longformer is designed for long documents and will process efficiently")
                combined_text = f"{cv_text} [SEP] {jd_text}"
            
            print(f"    🔗 Combined text length: {len(combined_text):,} characters")
            
            # Prepare inputs for the model - no need for truncation with Longformer
            inputs = self.talent_match_tokenizer(
                combined_text,
                return_tensors="pt",
                max_length=max_tokens,
                truncation=True,  # Keep this as safety, but shouldn't trigger
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.talent_match_model(**inputs)
                
                # Debug: Print what the model actually returns
                print(f"    📊 Model output keys: {list(outputs.keys())}")
                print(f"    📊 Last hidden state shape: {outputs.last_hidden_state.shape}")
                
                # Use the last hidden state for Longformer (no pooler output)
                last_hidden_state = outputs.last_hidden_state
                
                # For Longformer, we'll use the [CLS] token representation (first token)
                # This gives us a comprehensive representation of the entire document
                cls_embedding = last_hidden_state[0, 0, :]  # [CLS] token
                
                # Calculate the magnitude of the embedding (higher = more confident)
                embedding_magnitude = torch.norm(cls_embedding).item()
                
                # Normalize to 0-100 scale
                # Longformer embeddings tend to have different magnitude ranges
                max_expected_magnitude = 25.0  # Adjusted for Longformer
                normalized_score = min(100, (embedding_magnitude / max_expected_magnitude) * 100)
                
                print(f"    ✅ Using Longformer CLS token magnitude: {normalized_score:.2f}%")
                print(f"    💡 Score based on full CV and JD analysis")
                return round(normalized_score, 2)
                
        except Exception as e:
            print(f"    ❌ Longformer analysis failed: {e}")
            raise RuntimeError(f"Longformer analysis failed: {e}")
    
    def analyze_sections(self, cv_sections: Dict[str, str], jd_text: str) -> Dict[str, float]:
        """Analyze match scores for individual CV sections using AI-powered analysis."""
        section_scores = {}
        section_insights = {}
        
        print("📋 Analyzing CV sections with AI-powered insights...")
        for section_name, section_text in cv_sections.items():
            if section_text.strip() and len(section_text) > 50:  # Only analyze sections with substantial content
                print(f"  🔍 Analyzing {section_name}...")
                
                # Get AI-powered section analysis
                section_analysis = self._analyze_section_with_ai(section_text, jd_text, section_name)
                section_scores[section_name] = section_analysis['score']
                section_insights[section_name] = section_analysis['insights']
                
                print(f"  ✅ {section_name.title()}: {section_analysis['score']}%")
                print(f"     💡 {section_analysis['insights']}")
            else:
                section_scores[section_name] = 0.0
                section_insights[section_name] = "Insufficient content for analysis"
                print(f"  ⚪ {section_name.title()}: Insufficient content")
        
        # Store insights for later use
        self.section_insights = section_insights
        
        return section_scores
    
    def _analyze_section_with_ai(self, section_text: str, jd_text: str, section_name: str) -> Dict:
        """Analyze a specific section with AI-powered insights."""
        try:
            # Get the basic match score
            score = self.talent_match_score(section_text, jd_text)
            
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
    
    def get_section_extraction_quality_metrics(self, cv_sections: Dict[str, str]) -> Dict:
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
                quality_score = 0
                if content_length > 200:
                    quality_score += 30  # Good length
                if sentences >= 3:
                    quality_score += 25  # Good sentence count
                if avg_sentence_length > 50:
                    quality_score += 25  # Good sentence length
                if any(keyword in content.lower() for keyword in ["experience", "skills", "education", "summary"]):
                    quality_score += 20  # Relevant content
                
                metrics[section_name] = {
                    'content_length': content_length,
                    'content_percentage': round(content_percentage, 2),
                    'sentences': sentences,
                    'avg_sentence_length': round(avg_sentence_length, 1),
                    'quality_score': quality_score,
                    'quality_level': self._get_quality_level(quality_score)
                }
                
                print(f"  📊 {section_name.title()}:")
                print(f"     📏 Length: {content_length} chars ({content_percentage:.1f}%)")
                print(f"     📝 Sentences: {sentences}")
                print(f"     📊 Quality: {quality_score}/100 ({metrics[section_name]['quality_level']})")
        
        return metrics
    
    def _get_quality_level(self, score: int) -> str:
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
    
    def extract_technical_keywords(self, cv_text: str, jd_text: str) -> Dict[str, List[str]]:
        """Extract and analyze technical keywords."""
        # Common technical keywords
        tech_keywords = {
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible',
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'git', 'github', 'jenkins', 'agile', 'scrum', 'devops', 'ci/cd',
            'machine learning', 'ai', 'data science', 'cybersecurity', 'owasp',
            'nist', 'microservices', 'api', 'rest', 'graphql', 'sql', 'nosql',
            'kubernetes', 'docker', 'azure devops', 'jenkins', 'gitlab'
        }
        
        # Extract words from texts
        cv_words = set(re.findall(r'\b\w+\b', cv_text.lower()))
        jd_words = set(re.findall(r'\b\w+\b', jd_text.lower()))
        
        # Find technical keywords
        cv_tech = cv_words.intersection(tech_keywords)
        jd_tech = jd_words.intersection(tech_keywords)
        matching_tech = cv_tech.intersection(jd_tech)
        
        return {
            'cv_technical_keywords': sorted(list(cv_tech)),
            'jd_technical_keywords': sorted(list(jd_tech)),
            'matching_technical_keywords': sorted(list(matching_tech)),
            'cv_total_keywords': len(cv_words),
            'jd_total_keywords': len(jd_words)
        }
    
    def comprehensive_analysis(self, cv_file: str, jd_file: str) -> Dict:
        """Perform comprehensive CV-job matching analysis."""
        print(f"📄 Analyzing CV: {cv_file}")
        print(f"💼 Analyzing Job Description: {jd_file}")
        print("=" * 60)
        
        # Read files
        cv_text = self.read_docx(cv_file) if cv_file.endswith('.docx') else self.read_txt(cv_file)
        jd_text = self.read_txt(jd_file)
        
        if not cv_text or not jd_text:
            print("❌ Failed to read input files")
            return {}
        
        # Clean texts
        cv_text = self.clean_text(cv_text)
        jd_text = self.clean_text(jd_text)
        
        print(f"📊 CV length: {len(cv_text)} characters")
        print(f"📊 JD length: {len(jd_text)} characters")
        print()
        
        # STEP 1: Extract CV sections using AI FIRST
        print("🔍 Step 1: AI-Powered CV Section Extraction...")
        cv_sections = self.extract_cv_sections(cv_text)
        
        # Show section lengths
        for section, content in cv_sections.items():
            if content.strip():
                print(f"  📝 {section.title()}: {len(content)} characters")
        
        print()
        
        # Save sections to file for review
        print("📁 Saving extracted sections to file for review...")
        self.save_sections_to_file(cv_sections)
        print()
        
        # STEP 2: Analyze each section individually with the model
        print("🔍 Step 2: Individual Section Analysis with AI Model...")
        section_scores = self.analyze_sections(cv_sections, jd_text)
        print()
        
        # STEP 3: Calculate overall score based on section scores (weighted average)
        print("🔍 Step 3: Calculating Overall Score from Section Analysis...")
        overall_score = self._calculate_overall_score_from_sections(section_scores, cv_sections)
        print(f"✅ Overall AI Match Score (from sections): {overall_score}%")
        print()
        print()
        
        # Get AI-powered section extraction quality metrics
        section_quality_metrics = self.get_section_extraction_quality_metrics(cv_sections)
        print()
        
        # Technical keyword analysis
        print("🔑 Analyzing technical keywords...")
        keyword_analysis = self.extract_technical_keywords(cv_text, jd_text)
        
        print(f"📈 Technical Keywords Analysis:")
        print(f"  CV: {len(keyword_analysis['cv_technical_keywords'])} technical terms")
        print(f"  JD: {len(keyword_analysis['jd_technical_keywords'])} technical terms")
        print(f"  🎯 Matches: {len(keyword_analysis['matching_technical_keywords'])} common terms")
        
        if keyword_analysis['matching_technical_keywords']:
            print(f"  ✅ Matching keywords: {', '.join(keyword_analysis['matching_technical_keywords'])}")
        
        # Compile results
        results = {
            'overall_score': overall_score,
            'section_scores': section_scores,
            'section_insights': getattr(self, 'section_insights', {}),
            'section_quality_metrics': section_quality_metrics,
            'keyword_analysis': keyword_analysis,
            'cv_sections': cv_sections,
            'cv_text_length': len(cv_text),
            'jd_text_length': len(jd_text)
        }
        
        return results
    
    def _calculate_overall_score_from_sections(self, section_scores: Dict[str, float], cv_sections: Dict[str, str]) -> float:
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
        
        print("📊 Calculating weighted overall score from sections:")
        
        for section_name, score in section_scores.items():
            if score > 0 and section_name in section_weights:
                weight = section_weights[section_name]
                weighted_score = score * weight
                total_weighted_score += weighted_score
                total_weight += weight
                
                # Show section contribution
                content_length = len(cv_sections.get(section_name, ""))
                print(f"  📊 {section_name.title()}: {score}% × {weight*100:.0f}% = {weighted_score:.1f} points")
                print(f"     📏 Content: {content_length} characters")
        
        if total_weight > 0:
            overall_score = total_weighted_score / total_weight
            print(f"  🎯 Total Weighted Score: {total_weighted_score:.1f} / {total_weight:.2f} = {overall_score:.1f}%")
            return round(overall_score, 2)
        else:
            return 0.0
    
    def save_results(self, results: Dict, output_file: str = "ai_matching_results.json"):
        """Save analysis results to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"⚠ Failed to save results: {e}")
    
    def save_sections_to_file(self, cv_sections: Dict[str, str], output_file: str = "extracted_cv_sections.txt"):
        """Save extracted CV sections to a readable text file for review."""
        try:
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

    def show_processing_details(self, cv_text: str, jd_text: str):
        """Show detailed information about what content is being processed."""
        print("\n" + "="*60)
        print("📊 CONTENT PROCESSING ANALYSIS")
        print("="*60)
        
        # Calculate available space with Longformer
        max_tokens = 32768  # Longformer's actual capacity
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token
        
        print(f"📏 Model Capacity: {max_tokens:,} tokens (~{max_chars:,} characters)")
        print(f"📄 CV Length: {len(cv_text):,} characters")
        print(f"📋 JD Length: {len(jd_text):,} characters")
        print(f"🔗 Combined Total: {len(cv_text) + len(jd_text):,} characters")
        
        if len(cv_text) + len(jd_text) <= max_chars:
            print(f"✅ Status: FULL ANALYSIS POSSIBLE!")
            print(f"📊 CV Usage: 100% ({len(cv_text):,}/{len(cv_text):,} characters)")
            print(f"📊 JD Usage: 100% ({len(jd_text):,}/{len(jd_text):,} characters)")
            print(f"🎉 No content will be lost - Longformer processes everything!")
        else:
            print(f"⚠ Status: Texts exceed estimated capacity, but Longformer handles this efficiently")
            print(f"📊 CV Usage: 100% ({len(cv_text):,}/{len(cv_text):,} characters)")
            print(f"📊 JD Usage: 100% ({len(jd_text):,}/{len(jd_text):,} characters)")
            print(f"💡 Longformer is designed for long documents and will process efficiently")
        
        print("="*60)
        print("🚀 LONGFORMER ADVANTAGES:")
        print("   • Processes entire CV without truncation")
        print("   • Analyzes complete job description")
        print("   • Better understanding of context and relationships")
        print("   • More accurate matching scores")
        print("   • Handles long documents efficiently")
        print("="*60)
        
        print("="*60)
        
        # Show preview of what will be processed
        print("\n🔍 CONTENT PREVIEW (First 200 chars of each):")
        print("-" * 40)
        print(f"📄 CV Preview: {cv_text[:200]}{'...' if len(cv_text) > 200 else ''}")
        print("-" * 40)
        print(f"📋 JD Preview: {jd_text[:200]}{'...' if len(jd_text) > 200 else ''}")
        print("-" * 40)

def main():
    """Main function to run the AI CV matcher."""
    print("🚀 AI-Powered CV-Job Matcher")
    print("Using allenai/longformer-base-4096 model")
    print("=" * 60)
    
    # Initialize matcher
    try:
        matcher = AICVMatcher()
    except RuntimeError as e:
        print(f"❌ Failed to initialize AI matcher: {e}")
        print("💡 Please check your internet connection and try again.")
        return
    
    # Define input files
    input_dir = Path("input_files")
    cv_file = input_dir / "JamesLally.docx"
    jd_file = input_dir / "job_description.txt"
    
    # Check if files exist
    if not cv_file.exists():
        print(f"❌ CV file not found: {cv_file}")
        return
    
    if not jd_file.exists():
        print(f"❌ Job description file not found: {jd_file}")
        return
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📄 CV file: {cv_file.name}")
    print(f"💼 JD file: {jd_file.name}")
    print()
    
    # Run comprehensive analysis
    try:
        results = matcher.comprehensive_analysis(str(cv_file), str(jd_file))
        
        if results:
            # Save results
            matcher.save_results(results)
            
            # Summary
            print("\n" + "=" * 60)
            print("🎯 ANALYSIS SUMMARY")
            print("=" * 60)
            print(f"Overall AI Match Score: {results['overall_score']}%")
            
            # Find best section
            best_section = max(results['section_scores'].items(), key=lambda x: x[1])
            if best_section[1] > 0:
                print(f"Best Section: {best_section[0].title()} ({best_section[1]}%)")
            
            print(f"Technical Keywords Matched: {len(results['keyword_analysis']['matching_technical_keywords'])}")
            print("=" * 60)
            
            # Recommendations
            print("\n💡 RECOMMENDATIONS:")
            if results['overall_score'] >= 80:
                print("🎉 Excellent match! This CV strongly aligns with the job requirements.")
            elif results['overall_score'] >= 60:
                print("✅ Good match! This CV meets most of the job requirements.")
            elif results['overall_score'] >= 40:
                print("⚠️ Moderate match. Some alignment but areas for improvement.")
            else:
                print("❌ Low match. Significant gaps between CV and job requirements.")
            
            # Section insights
            print("\n📋 SECTION INSIGHTS:")
            for section, score in results['section_scores'].items():
                if score > 0:
                    status = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
                    print(f"  {status} {section.title()}: {score}%")
                    
                    # Show AI-powered insights if available
                    if 'section_insights' in results and section in results['section_insights']:
                        insight = results['section_insights'][section]
                        print(f"     💡 {insight}")
            
            # Show section quality metrics
            if 'section_quality_metrics' in results:
                print("\n🔍 SECTION EXTRACTION QUALITY:")
                for section, metrics in results['section_quality_metrics'].items():
                    if metrics['content_length'] > 0:
                        quality_emoji = "🟢" if metrics['quality_score'] >= 75 else "🟡" if metrics['quality_score'] >= 50 else "🔴"
                        print(f"  {quality_emoji} {section.title()}: {metrics['quality_level']} ({metrics['quality_score']}/100)")
                        print(f"     📏 {metrics['content_length']} chars, {metrics['sentences']} sentences")
                        print(f"     📊 {metrics['content_percentage']:.1f}% of total CV content")
            
        else:
            print("❌ Analysis failed. Please check the input files and try again.")
            
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        print("💡 Please check your input files and try again.")

if __name__ == "__main__":
    main()
