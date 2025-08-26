import docx
import re
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import spacy
from typing import Dict, List, Set, Tuple

# --------------------------
# NER Setup
# --------------------------

def load_spacy_model():
    """Load spaCy model for NER. Falls back to smaller model if transformer model not available."""
    try:
        # Try to load the transformer model first
        nlp = spacy.load("en_core_web_trf")
        print("✓ Loaded spaCy transformer model (en_core_web_trf)")
    except OSError:
        try:
            # Fallback to medium model
            nlp = spacy.load("en_core_web_md")
            print("✓ Loaded spaCy medium model (en_core_web_md)")
        except OSError:
            # Fallback to small model
            nlp = spacy.load("en_core_web_sm")
            print("✓ Loaded spaCy small model (en_core_web_sm)")
    return nlp

# --------------------------
# NER Functions
# --------------------------

def extract_entities(text: str, nlp) -> Dict[str, Set[str]]:
    """Extract named entities from text using spaCy NER."""
    doc = nlp(text)
    entities = defaultdict(set)
    
    # Define entity categories we care about
    entity_mapping = {
        'ORG': 'organizations',      # Companies, institutions
        'PERSON': 'people',          # Names
        'GPE': 'locations',          # Countries, cities
        'PRODUCT': 'technologies',    # Software, tools, products
        'WORK_OF_ART': 'skills',     # Skills, certifications
        'LAW': 'certifications',     # Legal entities, certifications
        'LANGUAGE': 'languages',     # Programming languages
        'DATE': 'dates',             # Dates, time periods
        'CARDINAL': 'numbers',       # Numbers, quantities
        'MONEY': 'salary',           # Salary, budget
    }
    
    for ent in doc.ents:
        category = entity_mapping.get(ent.label_, 'other')
        # Clean and normalize entity text
        clean_entity = ent.text.strip().lower()
        if len(clean_entity) > 2:  # Filter out very short entities
            entities[category].add(clean_entity)
    
    # Extract additional technical terms and skills
    technical_terms = extract_technical_terms(doc)
    entities['technologies'].update(technical_terms)
    
    return entities

def extract_technical_terms(doc) -> Set[str]:
    """Extract technical terms, skills, and technologies using POS patterns."""
    technical_terms = set()
    
    # Look for noun phrases that might be technologies/skills
    for chunk in doc.noun_chunks:
        text = chunk.text.lower().strip()
        # Filter for likely technical terms
        if (len(text) > 2 and 
            any(char.isupper() for char in chunk.text) or  # Has uppercase (e.g., Python, React)
            any(char.isdigit() for char in text) or        # Has numbers (e.g., Python 3.8)
            text in common_tech_terms):                    # Known tech terms
            technical_terms.add(text)
    
    # Look for individual tokens that might be skills
    for token in doc:
        if (token.pos_ in ['NOUN', 'PROPN'] and 
            len(token.text) > 2 and
            token.text.lower() in common_tech_terms):
            technical_terms.add(token.text.lower())
    
    return technical_terms

# Common technical terms and skills
common_tech_terms = {
    # Programming languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'swift', 'kotlin',
    'php', 'ruby', 'scala', 'r', 'matlab', 'perl', 'bash', 'powershell', 'sql', 'html', 'css',
    
    # Frameworks and libraries
    'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'express', 'laravel',
    'asp.net', 'jquery', 'bootstrap', 'tailwind', 'material-ui', 'redux', 'vuex', 'graphql',
    
    # Databases
    'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'dynamodb', 'sqlite', 'oracle',
    
    # Cloud platforms
    'aws', 'azure', 'gcp', 'heroku', 'digitalocean', 'kubernetes', 'docker', 'terraform',
    
    # Tools and methodologies
    'git', 'jenkins', 'ci/cd', 'agile', 'scrum', 'kanban', 'devops', 'microservices',
    'rest', 'api', 'json', 'xml', 'yaml', 'docker', 'kubernetes', 'terraform', 'ansible',
    
    # Skills and concepts
    'machine learning', 'ai', 'data science', 'big data', 'analytics', 'statistics',
    'testing', 'tdd', 'bdd', 'unit testing', 'integration testing', 'performance testing',
    'security', 'authentication', 'authorization', 'oauth', 'jwt', 'ssl', 'tls'
}

def extract_skills_from_entities(entities: Dict[str, Set[str]]) -> Set[str]:
    """Extract skills and technologies from entities."""
    skills = set()
    
    # Combine relevant entity categories
    skills.update(entities.get('technologies', set()))
    skills.update(entities.get('skills', set()))
    skills.update(entities.get('certifications', set()))
    skills.update(entities.get('languages', set()))
    
    return skills

def extract_companies_from_entities(entities: Dict[str, Set[str]]) -> Set[str]:
    """Extract company names from entities."""
    return entities.get('organizations', set())

def extract_locations_from_entities(entities: Dict[str, Set[str]]) -> Set[str]:
    """Extract locations from entities."""
    return entities.get('locations', set())

# --------------------------
# Enhanced Keyword Extraction
# --------------------------

def extract_keywords_with_ner(text: str, nlp) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Extract keywords using both traditional methods and NER."""
    # Traditional keyword extraction
    traditional_keywords = extract_keywords(text)
    
    # NER-based extraction
    entities = extract_entities(text, nlp)
    ner_skills = extract_skills_from_entities(entities)
    
    # Combine both approaches
    all_keywords = traditional_keywords.union(ner_skills)
    
    return all_keywords, entities

# --------------------------
# Enhanced Scoring
# --------------------------

def score_match_with_ner(cv_sections: Dict[str, List[str]], jd_keywords: Set[str], 
                         cv_entities: Dict[str, Set[str]], jd_entities: Dict[str, Set[str]]) -> Tuple[float, Dict, Set, Dict]:
    """Enhanced scoring that considers NER entities."""
    weights = {
        "Experience": 2.0,
        "Skills": 1.5,
        "Education": 1.0,
        "Summary": 1.2,
        "Other": 0.5,
    }

    matched = defaultdict(set)
    missing = set(jd_keywords)
    
    # Extract skills from CV entities
    cv_skills = extract_skills_from_entities(cv_entities)
    jd_skills = extract_skills_from_entities(jd_entities)
    
    # Score based on section matching
    total_score = 0
    max_score = 0

    for section, paragraphs in cv_sections.items():
        section_text = " ".join(paragraphs).lower()
        section_keywords = set(section_text.split())

        for kw in jd_keywords:
            if kw in section_keywords:
                matched[section].add(kw)
                total_score += weights.get(section, 1.0)
            max_score += weights.get(section, 1.0)

    # Bonus points for skill matches
    skill_matches = cv_skills.intersection(jd_skills)
    skill_bonus = len(skill_matches) * 0.5
    total_score += skill_bonus
    
    # Remove matched from missing
    for sec in matched.values():
        missing -= sec

    score_pct = round((total_score / max_score) * 100, 2) if max_score > 0 else 0
    
    # Additional insights
    insights = {
        'skill_matches': skill_matches,
        'cv_skills': cv_skills,
        'jd_skills': jd_skills,
        'companies': extract_companies_from_entities(cv_entities),
        'locations': extract_locations_from_entities(cv_entities)
    }
    
    return score_pct, matched, missing, insights

# --------------------------
# Helpers (existing functions)
# --------------------------

def read_docx(file_path):
    """Read text from a .docx CV file and return as list of paragraphs."""
    doc = docx.Document(file_path)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def read_txt(file_path):
    """Read text from a .txt job description file."""
    return Path(file_path).read_text()

def clean_text(text):
    """Lowercase and remove non-alphanumeric chars."""
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower())

def extract_keywords(text, min_len=3):
    """Extract keywords from text (simple heuristic)."""
    words = clean_text(text).split()
    stopwords = {
        "and", "the", "with", "for", "you", "are", "our", "but", "job", "role",
        "etc", "this", "that", "from", "into", "about", "what", "who", "when",
        "where", "how", "will", "have", "has", "had", "can", "all", "more"
    }
    return set([w for w in words if len(w) >= min_len and w not in stopwords])

def split_sections(paragraphs):
    """Split CV into sections based on headings."""
    sections = defaultdict(list)
    current_section = "Other"

    for p in paragraphs:
        low = p.lower()
        if "experience" in low:
            current_section = "Experience"
        elif "skills" in low:
            current_section = "Skills"
        elif "education" in low:
            current_section = "Education"
        elif "summary" in low or "profile" in low:
            current_section = "Summary"

        sections[current_section].append(p)

    return sections

def semantic_score(cv_text, jd_text):
    """Semantic similarity score using embeddings."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    cv_embedding = model.encode(cv_text, convert_to_tensor=True)
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    similarity = util.cos_sim(cv_embedding, jd_embedding).item()
    return round(similarity * 100, 2)

# --------------------------
# Main
# --------------------------

if __name__ == "__main__":
    # Load spaCy model
    print("Loading spaCy model...")
    nlp = load_spacy_model()
    
    # Update these paths
    cv_file = "input_files/JamesLally_b1.docx"   # your CV
    jd_file = "input_files/job_description.txt"  # save JD text here

    # Read files
    cv_paragraphs = read_docx(cv_file)
    cv_text = " ".join(cv_paragraphs)
    jd_text = read_txt(jd_file)

    # Extract sections & keywords with NER
    cv_sections = split_sections(cv_paragraphs)
    jd_keywords, jd_entities = extract_keywords_with_ner(jd_text, nlp)
    cv_entities = extract_entities(cv_text, nlp)

    # Enhanced ATS keyword scoring with NER
    ats_score, matched, missing, insights = score_match_with_ner(
        cv_sections, jd_keywords, cv_entities, jd_entities
    )

    # Semantic similarity scoring
    sem_score = semantic_score(cv_text, jd_text)

    # Output
    print("\n" + "="*60)
    print(f" Recruiter-style ATS Match Score: {ats_score}%")
    print(f" AI Semantic Match Score:        {sem_score}%")
    print("="*60)

    # Show matched keywords by section
    for section, kws in matched.items():
        print(f"\nMatched in {section} ({len(kws)}): {sorted(list(kws))}")

    # Show missing keywords
    print(f"\nMissing Keywords ({len(missing)}): {sorted(list(missing))}")
    
    # Show NER insights
    print("\n" + "="*60)
    print("NER INSIGHTS:")
    print("="*60)
    
    if insights['skill_matches']:
        print(f"\n✓ Skills Match ({len(insights['skill_matches'])}): {sorted(list(insights['skill_matches']))}")
    
    if insights['cv_skills']:
        print(f"\nCV Skills ({len(insights['cv_skills'])}): {sorted(list(insights['cv_skills']))}")
    
    if insights['jd_skills']:
        print(f"\nJob Required Skills ({len(insights['jd_skills'])}): {sorted(list(insights['jd_skills']))}")
    
    if insights['companies']:
        print(f"\nCompanies Mentioned: {sorted(list(insights['companies']))}")
    
    if insights['locations']:
        print(f"\nLocations Mentioned: {sorted(list(insights['locations']))}")
    
    print("="*60)
