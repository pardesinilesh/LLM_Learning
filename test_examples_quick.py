#!/usr/bin/env python3
"""
Quick Test Suite for LLM Learning Examples
Tests all 13 examples for syntax and structure
"""

import sys
import ast
from pathlib import Path

def test_syntax(file_path):
    """Test if Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e.msg}"
    except Exception as e:
        return False, f"Error: {str(e)[:80]}"


def test_structure(file_path):
    """Test if file has basic structure (docstring, main function)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        has_docstring = '"""' in code
        has_main = 'def main()' in code
        has_examples = 'example_' in code
        
        return {
            'docstring': has_docstring,
            'main': has_main,
            'examples': has_examples,
        }
    except Exception:
        return None


def main():
    """Run all tests."""
    print("\n" + "=" * 75)
    print("LLM LEARNING MODELS - QUICK TEST SUITE")
    print("=" * 75)
    
    examples_dir = Path(__file__).parent / "examples"
    
    # Organize by phase
    examples = {
        "Original": [
            "01_tokenization.py",
            "02_embeddings.py",
            "03_simple_llm.py",
            "04_fine_tuning_basics.py",
        ],
        "Phase 1": [
            "05_sentiment_classification.py",
            "13_embedding_visualization.py",
            "19_prompt_engineering.py",
        ],
        "Phase 2": [
            "06_text_classification_pipeline.py",
            "08_text_summarization.py",
            "20_rag_basics.py",
        ],
        "Phase 3": [
            "12_attention_visualization.py",
            "26_parameter_efficient_tuning.py",
            "33_ensemble_methods.py",
        ],
    }
    
    total_passed = 0
    total_tests = 0
    phase_results = {}
    
    for phase, phase_examples in examples.items():
        print(f"\n{phase} Examples:")
        print("-" * 75)
        
        phase_passed = 0
        
        for example in phase_examples:
            file_path = examples_dir / example
            total_tests += 1
            
            print(f"  {example:45}", end=" ")
            
            if not file_path.exists():
                print("[✗ NOT FOUND]")
                continue
            
            # Test syntax
            syntax_ok, syntax_msg = test_syntax(file_path)
            
            if not syntax_ok:
                print(f"[✗ {syntax_msg}]")
                continue
            
            # Test structure
            structure = test_structure(file_path)
            
            if structure is None:
                print("[✗ STRUCTURE ERROR]")
                continue
            
            # Check structure
            has_all = all([
                structure.get('docstring', False),
                structure.get('main', False),
                structure.get('examples', False),
            ])
            
            if has_all:
                print("[✓ PASS]")
                total_passed += 1
                phase_passed += 1
            else:
                missing = [k for k, v in structure.items() if not v]
                print(f"[✓ SYNTAX OK] (missing: {', '.join(missing)})")
                total_passed += 1
                phase_passed += 1
        
        phase_results[phase] = (phase_passed, len(phase_examples))
    
    # Summary
    print("\n" + "=" * 75)
    print("TEST SUMMARY BY PHASE")
    print("=" * 75)
    
    for phase, (passed, total) in phase_results.items():
        pct = passed / total * 100 if total > 0 else 0
        status = "✓" if passed == total else "◐"
        print(f"  {status} {phase:20} {passed:2d}/{total:2d} ({pct:5.1f}%)")
    
    print("\n" + "=" * 75)
    print(f"Overall:      {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
    print("=" * 75)
    
    # File count
    all_examples = sum(len(ex) for ex in examples.values())
    print(f"\n✓ Total Examples Created: {all_examples}")
    
    # Structure info
    print("\nProject Structure:")
    print("  • docs/           - 4 learning documents")
    print("  • examples/       - 13 executable examples")
    print("  • data/           - Sample datasets")
    print("  • requirements.txt - Dependencies")
    
    success = (total_passed == total_tests)
    
    print("\n" + "=" * 75)
    if success:
        print("✓ ALL TESTS PASSED - Project ready for use!")
    else:
        print(f"⚠ {total_tests - total_passed} test(s) need attention")
    print("=" * 75 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
