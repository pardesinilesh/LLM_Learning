"""
Utility functions for LLM Learning Examples

Helper functions used across multiple examples.
"""

def check_imports():
    """
    Check if all required imports are available.
    
    Returns:
        bool: True if all imports successful, False otherwise
    """
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'numpy': 'NumPy',
    }
    
    missing = []
    for module_name, display_name in required.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(f"{display_name} ({module_name})")
    
    if missing:
        print("ERROR: Missing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nRun: pip install -r requirements.txt")
        return False
    
    return True


def print_separator(title="", length=60):
    """
    Print a formatted separator line.
    
    Args:
        title: Optional title to include in separator
        length: Total length of separator
    """
    if title:
        print("\n" + "=" * length)
        print(f"  {title}")
        print("=" * length)
    else:
        print("=" * length)


def print_section(text, indent=0):
    """
    Print a formatted section header.
    
    Args:
        text: Section text
        indent: Number of spaces to indent
    """
    prefix = " " * indent
    print(f"{prefix}\n{prefix}{text}")


def truncate_string(text, max_length=100):
    """
    Truncate string to max length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated string
    """
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    return text


def format_table(rows, headers=None):
    """
    Format a table for printing.
    
    Args:
        rows: List of tuples, each tuple is a row
        headers: Optional list of header strings
        
    Returns:
        Formatted table string
    """
    if not rows:
        return ""
    
    num_cols = len(rows[0])
    col_widths = [0] * num_cols
    
    # Calculate column widths
    all_rows = [headers] if headers else []
    all_rows.extend(rows)
    
    for row in all_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Format rows
    result = []
    if headers:
        header_row = " | ".join(
            str(h).ljust(col_widths[i]) 
            for i, h in enumerate(headers)
        )
        result.append(header_row)
        result.append("-" * len(header_row))
    
    for row in rows:
        formatted_row = " | ".join(
            str(cell).ljust(col_widths[i])
            for i, cell in enumerate(row)
        )
        result.append(formatted_row)
    
    return "\n".join(result)


def get_device():
    """
    Get the device to use for models (CPU or GPU).
    
    Returns:
        str: 'cuda' if GPU available, 'cpu' otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except:
        pass
    return 'cpu'


def get_model_size(model):
    """
    Calculate total number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def format_number(num):
    """
    Format a number with thousands separator.
    
    Args:
        num: Number to format
        
    Returns:
        str: Formatted number
    """
    return f"{num:,}"


def get_tensor_memory(tensor):
    """
    Calculate memory usage of a tensor in MB.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        float: Memory usage in MB
    """
    return tensor.element_size() * tensor.nelement() / 1024 / 1024


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
