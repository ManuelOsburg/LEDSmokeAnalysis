import os
import re
import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution


@lru_cache(maxsize=None)
def get_ledsa_version() -> str:
    """
    Version of the ledsa code that is actually running, for the headers of the result files.

    importlib.metadata.version('ledsa') reports the installed distribution, which is not necessarily the code that
    runs: with PYTHONPATH pointing to a source checkout, an older installed ledsa reports its own version. Therefore
    the distribution version is only used if the distribution is installed at the location of this package.
    Otherwise the version is read from the pyproject.toml of the source checkout. If the package lies in a git
    repository, the commit is appended (with '-modified' if there are uncommitted changes in the package).

    :return: Version string, e.g. '0.10.0 (git c0bbf50)'.
    :rtype: str
    """
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    version = _get_distribution_version(package_dir) or _get_pyproject_version(package_dir) or 'unknown'
    commit = _get_git_commit(package_dir)
    if commit:
        version += f' (git {commit})'
    return version


def _get_distribution_version(package_dir: str):
    try:
        dist = distribution('ledsa')
    except PackageNotFoundError:
        return None
    installed_dir = os.path.abspath(str(dist.locate_file('ledsa')))
    if os.path.normcase(installed_dir) != os.path.normcase(package_dir):
        return None
    return dist.version


def _get_pyproject_version(package_dir: str):
    pyproject = os.path.join(os.path.dirname(package_dir), 'pyproject.toml')
    try:
        with open(pyproject, encoding='utf-8') as file:
            match = re.search(r'^version\s*=\s*"([^"]+)"', file.read(), re.M)
    except OSError:
        return None
    return match.group(1) if match else None


def _get_git_commit(package_dir: str):
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=package_dir, capture_output=True,
                                text=True, timeout=10)
        if commit.returncode != 0:
            return None
        status = subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=package_dir, capture_output=True,
                                text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    modified = '-modified' if status.returncode == 0 and status.stdout.strip() else ''
    return commit.stdout.strip() + modified
