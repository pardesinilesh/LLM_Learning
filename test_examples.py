#!/usr/bin/env python3
"""
Test Suite for LLM Learning Examples
Tests all 13 examples for syntax, imports, and basic functionality
"""

import sys
import importlib.util
from pathlib import Path

def load_and_test_module(file_path):
    """Load a Python module and check if it imports without errors."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        module = importlib.util.module_from_spec(spec)
        
        # Try to load the module
        spec.loader.exec_module(module)
        
        return True, "✓ Syntax OK"
    except SyntaxError as e:
        return False, f"✗ Syntax Error: {e}"
    except ImportError as e:
        return False, f"✗ Import Error: {e}"
    except Exception as e:
        return False, f"✗ Error: {str(e)[:100]}"


def test_all_examples():
    """Test all example files."""
    examples_dir = Path(__file__).parent / "examples"
    
    examples = {
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
        "Original": [
            "01_tokenization.py",
            "02_embeddings.py",
            "03_simple_llm.py",
            "04_fine_tuning_basics.py",
        ],
    }
    
    all_passed = True
    total_tests = 0
    passed_tests = 0
    
    print("\n" + "=" * 70)
    print("LLM LEARNING MODELS - TEST SUITE")
    print("=" * 70)
    
    for phase, phase_examples in examples.items():
        print(f"\n{phase} Examples:")
        print("-" * 70)
        
        for example in phase_examples:
            file_path = examples_dir / example
            total_tests += 1
            
            if not file_path.exists():
                print(f"  ✗ {example:45} [FILE NOT FOUND]")
                all_passed = False
                continue
            
            success, message = load_and_test_module(file_path)
            
            if success:
                passed_tests += 1
                status = "PASS"
                symbol = "✓"
            else:
                status = "FAIL"
                symbol = "✗"
                all_passed = False
            
            print(f"  {symbol} {example:45} [{status}]")
            if not success:
                print(f"      {message}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests:   {total_tests}")
    print(f"Passed:        {passed_tests}")
    print(f"Failed:        {total_tests - passed_tests}")
    print(f"Success Rate:  {passed_tests/total_tests*100:.1f}%")
    print("=" * 70)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - See details above")
        return 1


def test_imports():
    """Test if all required dependencies can be imported."""
    print("\n" + "=" * 70)
    print("DEPENDENCY CHECK")
    print("=" * 70)
    
    dependencies = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "transformers": "Hugging Face Transformers",
        "tokenizers": "Tokenizers Library",
        "tqdm": "TQDM Progress Bars",
        "requests": "Requests HTTP Library",
        "click": "Click CLI Framework",
    }
    
    all_available = True
    
    for module_name, display_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"  ✓ {display_name:40} {module_name}")
        except ImportError:
            print(f"  ✗ {display_name:40} {module_name}")
            all_available = False
    
    print("=" * 70)
    
    if not all_available:
        print("\n⚠ Missing dependencies! Install with:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies available!")
        return True


def test_file_structure():
    """Verify project file structure."""
    print("\n" + "=" * 70)
    print("FILE STRUCTURE CHECK")
    print("=" * 70)
    
    required_files = [
        "README.md",
        "requirements.txt",
        "examples/__init__.py",
        "examples/utils.py",
        "data/sentiment_examples.csv",
    ]
    
    required_dirs = [
        "docs",
        "examples",
        "data",
    ]
    
    base_path = Path(__file__).parent
    all_exist = True
    
    print("\nDirectories:")
    for dir_name in required_dirs:
        path = base_path / dir_name
        if path.exists():
            print(f"  ✓ {dir_name:40} exists")
        else:
            print(f"  ✗ {dir_name:40} MISSING")
            all_exist = False
    
    print("\nFiles:")
    for file_name in required_files:
        path = base_path / file_name
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {file_name:40} ({size:,} bytes)")
        else:
            print(f"  ✗ {file_name:40} MISSING")
            all_exist = False
    
    print("=" * 70)
    
    if all_exist:
        print("\n✓ File structure complete!")
        return True
    else:
        print("\n⚠ Some files are missing!")
        return False


def main():
    """Run all tests."""
    
    # Test file structure
    structure_ok = test_file_structure()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test all examples
    result = test_all_examples()
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"File Structure:  {'✓ OK' if structure_ok else '✗ ISSUES'}")
    print(f"Dependencies:    {'✓ OK' if imports_ok else '✗ MISSING'}")
    print(f"Example Tests:   {'✓ ALL PASS' if result == 0 else '✗ FAILURES'}")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    sys.exit(main())
