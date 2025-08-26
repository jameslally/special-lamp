import docx
import re
from pathlib import Path
from collections import defaultdict

# --------------------------
# Helpers
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

def score_match(cv_sections, jd_keywords):
    """Score based on weighted keyword matching."""
    weights = {
        "Experience": 2.0,
        "Skills": 1.5,
        "Education": 1.0,
        "Summary": 1.2,
        "Other": 0.5,
    }

    matched = defaultdict(set)
    missing = set(jd_keywords)

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

    # Remove matched from missing
    for sec in matched.values():
        missing -= sec

    score_pct = round((total_score / max_score) * 100, 2) if max_score > 0 else 0
    return score_pct, matched, missing

# --------------------------
# Main
# --------------------------

if __name__ == "__main__":
    # Update these paths
    cv_file = "JamesLally_v9.docx"   # your CV
    jd_file = "job_description.txt"  # save JD text here

    # Read files
    cv_paragraphs = read_docx(cv_file)
    jd_text = read_txt(jd_file)

    # Extract sections & keywords
    cv_sections = split_sections(cv_paragraphs)
    jd_keywords = extract_keywords(jd_text)

    # Score
    score, matched, missing = score_match(cv_sections, jd_keywords)

    # Output
    print("="*50)
    print(f" Recruiter-style Match Score: {score}%")
    print("="*50)

    for section, kws in matched.items():
        print(f"\nMatched in {section} ({len(kws)}): {sorted(list(kws))}")

    print(f"\nMissing Keywords ({len(missing)}): {sorted(list(missing))}")
    print("="*50)
    print("End of Report")