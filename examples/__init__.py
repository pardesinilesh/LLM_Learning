"""
LLM Learning Models - Practical Examples

A collection of runnable examples to learn how Large Language Models work.

Examples:
---------
01_tokenization.py   : Learn how text is converted to tokens
02_embeddings.py     : Understand how tokens become meaningful vectors
03_simple_llm.py     : Run text generation with a real LLM
04_fine_tuning_basics: Learn how to adapt models for specific tasks

Usage:
------
Run any example directly:
    python examples/01_tokenization.py
    python examples/02_embeddings.py
    etc.

Each example is self-contained and includes:
- Theory explanation
- Step-by-step code
- Printed results
- Key insights

Learn more in docs/ folder!
"""

from .utils import (
    check_imports,
    print_separator,
    print_section,
    truncate_string,
    format_table,
    get_device,
    get_model_size,
    format_number,
    get_tensor_memory,
)

__version__ = "1.0.0"
__all__ = [
    'check_imports',
    'print_separator',
    'print_section',
    'truncate_string',
    'format_table',
    'get_device',
    'get_model_size',
    'format_number',
    'get_tensor_memory',
]
