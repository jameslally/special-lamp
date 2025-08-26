"""
Optimized Main Script for Modular CV Matcher

Uses shared AI models and optimized section extraction for better performance.
"""

import logging
import warnings

# Suppress Transformers library logging
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from pathlib import Path
from cv_matcher.core.matcher import AICVMatcher


def main():
    """Run the optimized modular AI CV matcher."""
    print("🚀 AI-Powered CV-Job Matcher")
    print("=" * 60)
    
    # Initialize optimized matcher
    try:
        matcher = AICVMatcher()
    except RuntimeError as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Define input files
    input_dir = Path("input_files")
    cv_file = input_dir / "JamesLally.docx"
    jd_file = input_dir / "job_description.txt"
    
    # Validate files
    if not cv_file.exists():
        print(f"❌ CV file not found: {cv_file}")
        return
    
    if not jd_file.exists():
        print(f"❌ Job description file not found: {jd_file}")
        return
    
    print(f"📄 CV: {cv_file.name}")
    print(f"💼 JD: {jd_file.name}")
    print()
    
    # Run optimized analysis
    try:
        results = matcher.comprehensive_analysis(str(cv_file), str(jd_file))
        
        if results:
            # Save results
            matcher.save_results(results)
            
            # Display summary
            display_summary(results)
        else:
            print("❌ Analysis failed")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


def display_summary(results):
    """Display a clean summary of the analysis results."""
    print("\n" + "=" * 60)
    print("🎯 ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Overall score
    print(f"Overall Match Score: {results['overall_score']}%")
    
    # Best section
    if results['section_scores']:
        best_section = max(results['section_scores'].items(), key=lambda x: x[1])
        if best_section[1] > 0:
            print(f"Best Section: {best_section[0].title()} ({best_section[1]}%)")
    
    print("=" * 60)
    
    # Section scores
    print("\n📋 SECTION SCORES:")
    for section, score in results['section_scores'].items():
        if score > 0:
            status = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
            print(f"  {status} {section.title()}: {score}%")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    score = results['overall_score']
    if score >= 80:
        print("🎉 Excellent match! Strong alignment with job requirements.")
    elif score >= 60:
        print("✅ Good match! Meets most job requirements.")
    elif score >= 40:
        print("⚠️ Moderate match. Some alignment, areas for improvement.")
    else:
        print("❌ Low match. Significant gaps between CV and job requirements.")


if __name__ == "__main__":
    main()
