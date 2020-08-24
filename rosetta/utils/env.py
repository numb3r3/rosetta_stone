from typing import Optional, Any
import sys
import os
import importlib.util
import subprocess

# get environment information


def get_git_hash() -> str:

    def _decode_bytes(b: bytes) -> str:
        return b.decode('ascii')[:-1]

    try:
        is_git_repo = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL).stdout
    except FileNotFoundError:
        return ''

    if _decode_bytes(is_git_repo) == 'true':
        git_hash = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                  stdout=subprocess.PIPE).stdout
        return _decode_bytes(git_hash)
    else:
        print('No git info available in this directory')
        return ''


def get_args() -> list:
    return sys.argv


def get_environ(name: str, default: Optional[Any] = None) -> str:
    return os.environ.get(name, default)
