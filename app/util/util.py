
from llama_index.core.utils import print_text


def pretty_print(text: str, verbose: bool, step: str, pre_text: str | None = None, color: str | None = None) -> None:
        """Print the text if verbose mode is enabled."""
        if verbose:
            print_text(f"[{step}] ", color="blue")
            if pre_text:
                print_text(pre_text, end="\n")
            print_text(text, color=color, end="\n")
