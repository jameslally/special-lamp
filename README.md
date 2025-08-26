# 🚀 AI-Powered CV-Job Matcher

A sophisticated CV-job matching system that simulates how an Applicant Tracking System (ATS) might score your CV against a job description. Built with a clean, modular architecture that separates concerns and makes the codebase maintainable and extensible.

## ✨ Features

- **AI-Powered Analysis**: Uses advanced language models for intelligent CV section extraction and matching
- **Modular Architecture**: Clean separation of concerns for maintainability and extensibility
- **Multi-Format Support**: Parses `.docx` CVs and `.txt` job descriptions
- **Intelligent Section Extraction**: Automatically identifies CV sections (Experience, Skills, Education, Summary, Other)
- **AI-Driven Scoring**: Advanced semantic analysis with weighted scoring by section
- **Quality Metrics**: Comprehensive evaluation of extraction quality and content relevance
- **Debug Output**: Organized debug files in dedicated folder for easy review
- **Performance Optimized**: Shared AI models between components for efficiency

## 🏗️ Architecture Overview

The system is organized into distinct packages with clear separation of concerns:

```
cv_matcher/
├── __init__.py                 # Main package exports
├── core/                       # Core orchestration logic
│   ├── __init__.py
│   └── matcher.py             # Main AICVMatcher class
├── section_extraction/         # CV section extraction logic
│   ├── __init__.py
│   └── extractor.py           # CVSectionExtractor class
├── matching/                   # Section analysis and matching
│   ├── __init__.py
│   └── analyzer.py            # SectionAnalyzer class
└── utils/                      # Utility functions
    ├── __init__.py
    ├── file_handlers.py       # File I/O operations
    └── quality_metrics.py     # Quality analysis utilities
```

## 🔧 Key Components

### 1. **Core Matcher** (`cv_matcher.core.matcher`)
- **Purpose**: Orchestrates the entire CV matching process
- **Responsibilities**: 
  - Coordinates between components
  - Manages the analysis workflow
  - Provides high-level API
  - Saves results to debug folder

### 2. **Section Extractor** (`cv_matcher.section_extraction.extractor`)
- **Purpose**: Intelligently extracts CV sections using AI and pattern matching
- **Responsibilities**:
  - Identifies section headers using regex patterns
  - Uses semantic analysis for content classification
  - Respects CV structure while leveraging AI insights
  - Handles fallback classification when AI is uncertain
  - Fast pattern-based classification for performance

### 3. **Section Analyzer** (`cv_matcher.matching.analyzer`)
- **Purpose**: Analyzes individual CV sections against job descriptions using AI
- **Responsibilities**:
  - Calculates section-specific match scores using Longformer model
  - Generates intelligent insights for each section
  - Computes weighted overall scores
  - Provides semantic analysis of content relevance

### 4. **File Handlers** (`cv_matcher.utils.file_handlers`)
- **Purpose**: Manages file I/O operations
- **Responsibilities**:
  - Reads DOCX and TXT files
  - Preserves document structure
  - Cleans and normalizes text
  - Handles file validation

### 5. **Quality Metrics** (`cv_matcher.utils.quality_metrics`)
- **Purpose**: Analyzes section extraction quality
- **Responsibilities**:
  - Evaluates content quality scores
  - Provides detailed metrics
  - Saves sections to debug folder for review

## 🎯 Benefits of Modular Architecture

### ✅ **Separation of Concerns**
- **Section Extraction**: Focuses solely on identifying and categorizing CV content
- **Section Analysis**: Handles only the matching and scoring logic
- **File Operations**: Manages I/O without business logic
- **Quality Assessment**: Evaluates extraction quality independently

### ✅ **Maintainability**
- Each component can be modified without affecting others
- Clear interfaces between modules
- Easier to debug and test individual components
- Reduced code coupling

### ✅ **Extensibility**
- Easy to add new section types
- Simple to implement new analysis algorithms
- Can swap out AI models independently
- Easy to add new file formats

### ✅ **Testability**
- Each component can be unit tested in isolation
- Mock dependencies easily
- Clear input/output contracts
- Reduced test complexity

### ✅ **Reusability**
- Components can be used independently
- Easy to integrate into other systems
- Can be extended for different use cases
- Clear API boundaries

## 🚀 Installation

1. **Clone or download this project**
2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Basic Usage
```python
from cv_matcher import AICVMatcher

# Initialize the matcher
matcher = AICVMatcher()

# Run comprehensive analysis
results = matcher.comprehensive_analysis("cv.docx", "job_description.txt")

# Save results (automatically goes to debug folder)
matcher.save_results(results)
```

### Component-Level Usage
```python
from cv_matcher.section_extraction.extractor import CVSectionExtractor
from cv_matcher.matching.analyzer import SectionAnalyzer

# Use components independently
extractor = CVSectionExtractor()
analyzer = SectionAnalyzer()

# Extract sections
sections = extractor.extract_sections(cv_text)

# Analyze sections
scores = analyzer.analyze_sections(sections, jd_text)
```

### Command Line Usage
```bash
python main.py
```

## 🔍 Workflow

1. **File Reading**: `FileHandler` reads and cleans CV and JD files
2. **Section Extraction**: `CVSectionExtractor` identifies and categorizes CV sections using AI and pattern matching
3. **Section Analysis**: `SectionAnalyzer` scores each section against the JD using advanced language models
4. **Score Calculation**: Weighted overall score computed from section scores
5. **Quality Assessment**: `QualityMetrics` evaluates extraction quality
6. **Results Compilation**: All results combined into comprehensive output
7. **Debug Output**: Results and sections saved to organized debug folder

## 🎨 Design Patterns Used

- **Strategy Pattern**: Different AI models can be swapped
- **Factory Pattern**: Component initialization
- **Observer Pattern**: Quality metrics and insights
- **Template Method**: Analysis workflow
- **Dependency Injection**: Component dependencies

## 🔧 Configuration

Each component can be configured independently:

```python
# Custom model
matcher = AICVMatcher(model_name="custom-model-name")

# Access individual components
extractor = matcher.get_section_extractor()
analyzer = matcher.get_section_analyzer()
file_handler = matcher.get_file_handler()
```

## 📊 Output

The system provides comprehensive output including:
- **Section Scores**: Individual section match percentages
- **AI Insights**: Intelligent analysis of each section
- **Quality Metrics**: Extraction quality assessment
- **Overall Score**: Weighted composite score
- **Debug Files**: Organized output in `debug/` folder:
  - `extracted_cv_sections.txt` - Detailed CV section extraction results
  - `ai_matching_results.json` - Complete analysis results in JSON format

## 🗂️ Debug Output Organization

All debug and output files are automatically organized in a `debug/` folder:
- **Automatic Creation**: Folder is created if it doesn't exist
- **Organized Structure**: All output files in one location
- **Version Control**: Debug folder is gitignored by default
- **Easy Review**: Simple access to all analysis outputs

## 🚀 Future Enhancements

The modular architecture makes it easy to add:
- New AI models for different languages
- Additional section types
- Custom scoring algorithms
- New file formats
- Integration with external APIs
- Real-time analysis capabilities

## 🧪 Testing

Each component can be tested independently:

```python
# Test section extraction
def test_section_extraction():
    extractor = CVSectionExtractor()
    sections = extractor.extract_sections(test_cv_text)
    assert "summary" in sections
    assert "experience" in sections

# Test section analysis
def test_section_analysis():
    analyzer = SectionAnalyzer()
    scores = analyzer.analyze_sections(test_sections, test_jd_text)
    assert all(0 <= score <= 100 for score in scores.values())
```

## 📈 Performance Features

- **Parallel Processing**: Components can run independently
- **Model Sharing**: Single AI model instance shared across components
- **Memory Efficiency**: Components release resources when done
- **Scalability**: Easy to add worker processes for batch processing
- **Hybrid Approach**: Combines fast pattern matching with AI analysis

## 🔍 Section Scoring Weights

The system uses intelligent weighting for different CV sections:
- **Experience**: 35% - Most important for job matching
- **Skills**: 30% - Technical capabilities
- **Summary**: 15% - Overview and cultural fit
- **Education**: 15% - Qualifications
- **Other**: 5% - Additional information

## 📝 Supported File Formats

- **CV Files**: `.docx` (Word documents)
- **Job Descriptions**: `.txt` (Text files)
- **Output**: JSON results and formatted text files

This modular architecture transforms the CV matching system from a monolithic application into a flexible, maintainable, and extensible platform that can evolve with changing requirements and technologies.

