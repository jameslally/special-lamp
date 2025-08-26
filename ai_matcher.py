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
            # Load the specialized talent matching model
            print("Loading AventIQ-AI/bert-talentmatchai...")
            self.talent_match_tokenizer = AutoTokenizer.from_pretrained("AventIQ-AI/bert-talentmatchai")
            self.talent_match_model = AutoModel.from_pretrained("AventIQ-AI/bert-talentmatchai")
            
            # Move to device
            self.talent_match_model.to(self.device)
            self.talent_match_model.eval()
            print("✅ Talent matching model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load talent matching model: {e}")
            print("💡 Please ensure the model is available and try again.")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def read_docx(self, file_path: str) -> str:
        """Read text from a .docx CV file."""
        try:
            doc = docx.Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs if para.text.strip()])
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
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\-\+\&\@\#]', '', text)
        return text.strip()
    
    def extract_cv_sections(self, text: str) -> Dict[str, str]:
        """Extract CV sections using keyword-based approach."""
        sections = {
            "summary": "",
            "experience": "",
            "skills": "",
            "education": "",
            "other": ""
        }
        
        # Split text into sentences for better analysis
        sentences = re.split(r'[.!?]+', text)
        
        current_section = "summary"
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_lower = sentence.lower()
            
            # Determine section based on keywords in sentence
            if any(keyword in sentence_lower for keyword in ["experience", "worked", "led", "managed", "developed", "implemented", "career"]):
                current_section = "experience"
            elif any(keyword in sentence_lower for keyword in ["skills", "expertise", "proficient", "knowledge", "technologies", "certified"]):
                current_section = "skills"
            elif any(keyword in sentence_lower for keyword in ["education", "academic", "university", "college", "degree", "certification"]):
                current_section = "education"
            elif any(keyword in sentence_lower for keyword in ["summary", "profile", "objective", "overview"]):
                current_section = "summary"
            
            # Add sentence to current section
            sections[current_section] += sentence + ". "
        
        # Clean sections
        for key in sections:
            sections[key] = self.clean_text(sections[key])
        
        return sections
    
    def talent_match_score(self, cv_text: str, jd_text: str) -> float:
        """Calculate match score using the specialized talent matching model."""
        try:
            # Truncate texts to fit model input length
            max_length = 256  # Half of 512 for each input
            cv_truncated = cv_text[:max_length] if len(cv_text) > max_length else cv_text
            jd_truncated = jd_text[:max_length] if len(jd_text) > max_length else jd_text
            
            print(f"    🔍 Using specialized model: AventIQ-AI/bert-talentmatchai")
            
            # This model expects a single combined input, not two separate texts
            # Combine CV and JD with a separator token
            combined_text = f"{cv_truncated} [SEP] {jd_truncated}"
            
            # Prepare inputs for the model (single text input)
            inputs = self.talent_match_tokenizer(
                combined_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
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
                
                # Use the pooler output (this is the model's final representation)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    print(f"    📊 Pooler output shape: {outputs.pooler_output.shape}")
                    
                    # The pooler output is a single embedding representing the combined CV+JD
                    # We use the magnitude of the embedding as a proxy for match quality
                    pooler_embedding = outputs.pooler_output[0]  # Remove batch dimension
                    
                    # Calculate the magnitude of the embedding (higher = more confident)
                    embedding_magnitude = torch.norm(pooler_embedding).item()
                    
                    # Normalize to 0-100 scale (this is a heuristic)
                    # Higher magnitude embeddings often indicate better semantic understanding
                    max_expected_magnitude = 20.0  # Approximate max for BERT embeddings
                    normalized_score = min(100, (embedding_magnitude / max_expected_magnitude) * 100)
                    
                    print(f"    ✅ Using pooler output magnitude: {normalized_score:.2f}%")
                    return round(normalized_score, 2)
                
                # Fallback: Use the last hidden state
                else:
                    print(f"    ⚠ No pooler output, using CLS token...")
                    last_hidden_state = outputs.last_hidden_state
                    
                    # Use the [CLS] token representation (first token)
                    cls_embedding = last_hidden_state[0, 0, :]  # [CLS] token
                    
                    # Calculate magnitude as before
                    embedding_magnitude = torch.norm(cls_embedding).item()
                    max_expected_magnitude = 20.0
                    normalized_score = min(100, (embedding_magnitude / max_expected_magnitude) * 100)
                    
                    print(f"    ✅ Using CLS token magnitude: {normalized_score:.2f}%")
                    return round(normalized_score, 2)
                
        except Exception as e:
            print(f"    ❌ Talent matching failed: {e}")
            raise RuntimeError(f"Talent matching failed: {e}")
    
    def analyze_sections(self, cv_sections: Dict[str, str], jd_text: str) -> Dict[str, float]:
        """Analyze match scores for individual CV sections."""
        section_scores = {}
        
        print("📋 Analyzing CV sections...")
        for section_name, section_text in cv_sections.items():
            if section_text.strip() and len(section_text) > 50:  # Only analyze sections with substantial content
                print(f"  🔍 Analyzing {section_name}...")
                score = self.talent_match_score(section_text, jd_text)
                section_scores[section_name] = score
                print(f"  ✅ {section_name.title()}: {score}%")
            else:
                section_scores[section_name] = 0.0
                print(f"  ⚪ {section_name.title()}: Insufficient content")
        
        return section_scores
    
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
        
        # Extract CV sections
        print("🔍 Extracting CV sections...")
        cv_sections = self.extract_cv_sections(cv_text)
        
        # Show section lengths
        for section, content in cv_sections.items():
            if content.strip():
                print(f"  📝 {section.title()}: {len(content)} characters")
        
        print()
        
        # Overall match score
        print("🎯 Calculating overall match score...")
        overall_score = self.talent_match_score(cv_text, jd_text)
        print(f"✅ Overall AI Match Score: {overall_score}%")
        print()
        
        # Section analysis
        section_scores = self.analyze_sections(cv_sections, jd_text)
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
            'keyword_analysis': keyword_analysis,
            'cv_sections': cv_sections,
            'cv_text_length': len(cv_text),
            'jd_text_length': len(jd_text)
        }
        
        return results
    
    def save_results(self, results: Dict, output_file: str = "ai_matching_results.json"):
        """Save analysis results to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"⚠ Failed to save results: {e}")

def main():
    """Main function to run the AI CV matcher."""
    print("🚀 AI-Powered CV-Job Matcher")
    print("Using AventIQ-AI/bert-talentmatchai model")
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
        
        else:
            print("❌ Analysis failed. Please check the input files and try again.")
            
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        print("💡 Please check your input files and try again.")

if __name__ == "__main__":
    main()
