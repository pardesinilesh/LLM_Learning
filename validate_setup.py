#!/usr/bin/env python3
"""
Setup Validation Script

Checks if your environment is properly configured for running LLM examples.
Run this before starting to ensure everything is set up correctly.
"""

import sys
import os


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_check(condition, message):
    """Print a check mark or cross mark."""
    symbol = "✓" if condition else "✗"
    status = "OK" if condition else "FAILED"
    print(f"  [{symbol}] {message:45} {status}")
    return condition


def check_python_version():
    """Check Python version."""
    print_header("Python Version Check")
    
    version = sys.version_info
    required = (3, 8)
    
    actual_version = f"{version.major}.{version.minor}.{version.micro}"
    print(f"  Current version: Python {actual_version}")
    
    if (version.major, version.minor) >= required:
        print(f"  Required version: Python 3.8+")
        print_check(True, "Python version")
        return True
    else:
        print(f"  Required version: Python 3.8+")
        print_check(False, "Python version")
        return False


def check_required_packages():
    """Check if required packages are installed."""
    print_header("Required Packages Check")
    
    all_ok = True
    
    required_packages = {
        'torch': 'PyTorch (deep learning)',
        'transformers': 'Transformers (HuggingFace)',
        'numpy': 'NumPy (numerical computing)',
        'tokenizers': 'Tokenizers (fast tokenization)',
    }
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            version = sys.modules[package].__version__
            print_check(True, f"{description:30} v{version}")
        except ImportError:
            print_check(False, f"{description:30}")
            all_ok = False
    
    return all_ok


def check_optional_packages():
    """Check optional packages."""
    print_header("Optional Packages Check")
    
    optional_packages = {
        'tqdm': 'Progress bars',
        'requests': 'HTTP requests',
        'click': 'CLI utilities',
    }
    
    print("  (These are optional but recommended)\n")
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
            version = sys.modules[package].__version__
            print_check(True, f"{description:30} v{version}")
        except ImportError:
            print_check(False, f"{description:30}")


def check_directory_structure():
    """Check if directory structure is correct."""
    print_header("Directory Structure Check")
    
    required_dirs = {
        'docs': 'Documentation files',
        'examples': 'Example scripts',
    }
    
    required_files = {
        'README.md': 'README',
        'QUICKSTART.md': 'Quick start guide',
        'requirements.txt': 'Dependencies file',
    }
    
    all_ok = True
    
    print("  Checking directories:\n")
    for dir_name, description in required_dirs.items():
        exists = os.path.isdir(dir_name)
        print_check(exists, f"Directory: {dir_name:20} ({description})")
        all_ok = all_ok and exists
    
    print("\n  Checking files:\n")
    for file_name, description in required_files.items():
        exists = os.path.isfile(file_name)
        print_check(exists, f"File: {file_name:25} ({description})")
        all_ok = all_ok and exists
    
    return all_ok


def check_example_files():
    """Check if example files exist."""
    print_header("Example Files Check")
    
    examples = [
        ('examples/01_tokenization.py', 'Tokenization example'),
        ('examples/02_embeddings.py', 'Embeddings example'),
        ('examples/03_simple_llm.py', 'Text generation example'),
        ('examples/04_fine_tuning_basics.py', 'Fine-tuning example'),
    ]
    
    all_ok = True
    
    for file_path, description in examples:
        exists = os.path.isfile(file_path)
        print_check(exists, f"{description:30}")
        all_ok = all_ok and exists
    
    return all_ok


def check_doc_files():
    """Check if documentation files exist."""
    print_header("Documentation Files Check")
    
    docs = [
        ('docs/01_introduction.md', 'Introduction'),
        ('docs/02_fundamentals.md', 'Fundamentals'),
        ('docs/03_how_llms_work.md', 'How LLMs work'),
        ('docs/04_practical_guide.md', 'Practical guide'),
    ]
    
    all_ok = True
    
    for file_path, description in docs:
        exists = os.path.isfile(file_path)
        print_check(exists, f"{description:30}")
        all_ok = all_ok and exists
    
    return all_ok


def check_imports():
    """Try importing key modules."""
    print_header("Import Tests")
    
    all_ok = True
    
    test_imports = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('numpy', 'NumPy'),
    ]
    
    for module_name, display_name in test_imports:
        try:
            __import__(module_name)
            print_check(True, f"Import {display_name}")
        except Exception as e:
            print_check(False, f"Import {display_name} ({str(e)[:20]}...)")
            all_ok = False
    
    return all_ok


def check_write_permissions():
    """Check if we can write files."""
    print_header("Write Permissions Check")
    
    test_file = '.write_test'
    
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print_check(True, "Write to current directory")
        return True
    except Exception as e:
        print_check(False, f"Write to current directory ({str(e)})")
        return False


def main():
    """Run all checks."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  LLM Learning Models - Setup Validation  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = {
        'Python Version': check_python_version(),
        'Required Packages': check_required_packages(),
        'Directory Structure': check_directory_structure(),
        'Example Files': check_example_files(),
        'Documentation': check_doc_files(),
        'Imports': check_imports(),
        'Write Permissions': check_write_permissions(),
    }
    
    check_optional_packages()
    
    # Summary
    print_header("Summary")
    
    all_passed = all(results.values())
    
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check_name:30} {status}")
    
    print()
    
    if all_passed:
        print("✓ All checks passed! You're ready to start learning.")
        print("\nNext steps:")
        print("  1. Read QUICKSTART.md")
        print("  2. Run: python examples/01_tokenization.py")
        print("  3. Enjoy learning about LLMs!")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("  - Run: pip install -r requirements.txt")
        print("  - Check your internet connection")
        print("  - Ensure you're in the correct directory")
        return 1


if __name__ == "__main__":
    sys.exit(main())
