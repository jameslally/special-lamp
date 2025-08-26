# special-lamp
shine light on the CV scoring

# ATS Matcher

A simple Python tool that simulates how an Applicant Tracking System (ATS) might score your CV against a job description.  
It parses your CV, extracts job description keywords, applies weighted scoring by section (Experience, Skills, Education, etc.), and produces a recruiter-style scorecard.

## Features
- Parses `.docx` CVs and `.txt` job descriptions
- Splits CV into sections (Experience, Skills, Education, Summary, Other)
- Weighted keyword scoring:
  - Experience: 2.0x
  - Skills: 1.5x
  - Education: 1.0x
  - Summary: 1.2x
  - Other: 0.5x
- Outputs:
  - Recruiter-style match score (percentage)
  - Matched keywords (grouped by section)
  - Missing keywords

## Installation

1. Clone or download this project.
2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows


## Create a virtual environment

This keeps your dependencies clean and avoids conflicts.

    python -m venv venv

Activate it
    venv\Scripts\activate
You’ll know it worked if (venv) shows up in your terminal prompt.
.. may need Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

