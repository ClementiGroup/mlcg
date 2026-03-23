RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def torch_compile_waring(e):
    return f"""
{RED}{BOLD}
============================================================
!!! DANGER WARNING: torch.compile FALLBACK ACTIVATED !!!
============================================================{RESET}

{YELLOW}torch.compile failed with error: 
-----------------------------------------------------------{RESET}
{e}
{YELLOW}
-----------------------------------------------------------

The model is now running with `torch._dynamo.config.suppress_errors = True`, which
is {BOLD}NOT recommended{RESET} {YELLOW}for production or benchmarking purposes.

Please check the following:
- Any code incompatible with torch.compile should be
  marked using the {BOLD}@torch.compile.disable{RESET} {YELLOW}decorator.
- If there are device or environment issues (e.g., no
  writable home directory for Triton cache), fix them.
- Performance may be dramatically reduced in this mode.{RESET}

{RED}{BOLD}============================================================{RESET}
"""


def force_torch_compile_warning():
    return f"""
{RED}
Forcing compilation with only one structure in the batch:{RESET}
{YELLOW}this may expose known issues with scatter operations and may produce invalid outputs or NaN gradients.{RESET}
"""
