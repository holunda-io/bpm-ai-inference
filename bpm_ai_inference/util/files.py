import glob
import os


def find_file(path: str, filename_pattern: str):
    """
    Find a file with the pattern below the given path.
    Return the first occurrence if there are multiple matches.

    Returns:
        str: The path to the first matching file, or None if no file is found.
    """
    search_pattern = os.path.join(path, "**", filename_pattern)
    matching_files = glob.glob(search_pattern, recursive=True)
    return matching_files[0] if matching_files else None