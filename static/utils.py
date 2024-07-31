import os

figures_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')


def inverse_slash_path(path: str):
    """
    This function takes a string representing a file path and switches all slashes ("/") to backslashes ("\\") and vice versa.

    Parameters:
    path (str): The file path as a string.

    Returns:
    str: The file path with slashes and backslashes inverted.

    Examples:
    # >>> inverse_slash_path("C:/Users/User/Documents")
    'C:\\Users\\User\\Documents'

    # >>> inverse_slash_path("C:\\Users\\User\\Documents")
    'C:/Users/User/Documents'
    """
    if '/' in path:
        return path.replace('/', '\\')
    elif '\\' in path:
        return path.replace('\\', '/')
    return path

