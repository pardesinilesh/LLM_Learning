#!/usr/bin/env python3
"""
Test Suite for LLM Learning Examples

Quick tests to verify all examples can be loaded and run
without actually downloading large models (for CI/CD purposes).
"""

import sys
import os


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import torch
        print("  ✓ torch")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    try:
        import transformers
        print("  ✓ transformers")
    except ImportError as e:
        print(f"  ✗ transformers: {e}")
        return False
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    return True


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'README.md',
        'QUICKSTART.md',
        'requirements.txt',
        'docs/01_introduction.md',
        'docs/02_fundamentals.md',
        'docs/03_how_llms_work.md',
        'docs/04_practical_guide.md',
        'examples/01_tokenization.py',
        'examples/02_embeddings.py',
        'examples/03_simple_llm.py',
        'examples/04_fine_tuning_basics.py',
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist


def test_syntax():
    """Test Python syntax of all examples."""
    print("\nTesting Python syntax...")
    
    import py_compile
    
    examples = [
        'examples/01_tokenization.py',
        'examples/02_embeddings.py',
        'examples/03_simple_llm.py',
        'examples/04_fine_tuning_basics.py',
    ]
    
    all_valid = True
    for example in examples:
        try:
            py_compile.compile(example, doraise=True)
            print(f"  ✓ {example}")
        except py_compile.PyCompileError as e:
            print(f"  ✗ {example}: {e}")
            all_valid = False
    
    return all_valid


def test_documentation():
    """Test that documentation files are readable."""
    print("\nTesting documentation...")
    
    docs = [
        'README.md',
        'QUICKSTART.md',
        'docs/01_introduction.md',
        'docs/02_fundamentals.md',
        'docs/03_how_llms_work.md',
        'docs/04_practical_guide.md',
    ]
    
    all_readable = True
    for doc in docs:
        try:
            with open(doc, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 100:
                    print(f"  ✓ {doc} ({len(content)} chars)")
                else:
                    print(f"  ✗ {doc} (too short)")
                    all_readable = False
        except Exception as e:
            print(f"  ✗ {doc}: {e}")
            all_readable = False
    
    return all_readable


def test_example_structure():
    """Test that examples have required structure."""
    print("\nTesting example structure...")
    
    examples = [
        'examples/01_tokenization.py',
        'examples/02_embeddings.py',
        'examples/03_simple_llm.py',
        'examples/04_fine_tuning_basics.py',
    ]
    
    all_ok = True
    for example in examples:
        try:
            with open(example, 'r') as f:
                content = f.read()
                
                checks = [
                    ('Has docstring', '"""' in content or "'''" in content),
                    ('Has main section', 'if __name__ == "__main__"' in content),
                    ('Has learning objective', 'LEARNING OBJECTIVE' in content),
                    ('Has examples', 'example_' in content),
                    ('Has print statements', 'print(' in content),
                ]
                
                example_ok = True
                for check_name, check_result in checks:
                    if check_result:
                        print(f"  ✓ {example}: {check_name}")
                    else:
                        print(f"  ✗ {example}: {check_name}")
                        example_ok = False
                
                all_ok = all_ok and example_ok
        except Exception as e:
            print(f"  ✗ {example}: {e}")
            all_ok = False
    
    return all_ok


def test_data_files():
    """Test that data files exist."""
    print("\nTesting data files...")
    
    data_files = [
        'data/sentiment_examples.csv',
    ]
    
    all_exist = True
    for data_file in data_files:
        if os.path.isfile(data_file):
            try:
                with open(data_file, 'r') as f:
                    lines = f.readlines()
                    print(f"  ✓ {data_file} ({len(lines)} lines)")
            except Exception as e:
                print(f"  ✗ {data_file}: {e}")
                all_exist = False
        else:
            print(f"  ✗ {data_file} (missing)")
            all_exist = False
    
    return all_exist


def test_github_workflows():
    """Test GitHub Actions workflow files."""
    print("\nTesting GitHub Actions...")
    
    workflow_file = '.github/workflows/test.yml'
    
    if os.path.isfile(workflow_file):
        try:
            with open(workflow_file, 'r') as f:
                content = f.read()
                
                checks = [
                    ('Has python setup', 'python-version' in content),
                    ('Has dependencies', 'pip install' in content),
                    ('Has tests', 'test' in content.lower()),
                ]
                
                all_ok = True
                for check_name, check_result in checks:
                    if check_result:
                        print(f"  ✓ {workflow_file}: {check_name}")
                    else:
                        print(f"  ✗ {workflow_file}: {check_name}")
                        all_ok = False
                
                return all_ok
        except Exception as e:
            print(f"  ✗ {workflow_file}: {e}")
            return False
    else:
        print(f"  ✗ {workflow_file} (missing)")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("  LLM Learning Models - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Python Syntax", test_syntax),
        ("Documentation", test_documentation),
        ("Example Structure", test_example_structure),
        ("Data Files", test_data_files),
        ("GitHub Actions", test_github_workflows),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\nError running {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  [{symbol}] {test_name:30} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓ ALL TESTS PASSED!")
        print("  Ready for development and deployment")
        return 0
    else:
        print("  ✗ SOME TESTS FAILED")
        print("  Please fix issues before deploying")
        return 1


if __name__ == "__main__":
    sys.exit(main())
