"""Ensure correct `frozen: x.x.x` comments in `pre-commit-config.yaml`."""


from __future__ import annotations

import argparse
import asyncio
import enum
import logging
import os
import re
import sqlite3
import subprocess
import tempfile
from asyncio import create_subprocess_exec, gather
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import lru_cache, partial
from io import StringIO
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional, Tuple, cast

from ruamel.yaml import YAML
from ruamel.yaml.util import load_yaml_guess_indent

try:
    from rich.console import Console
    from rich.logging import RichHandler

    COLOUR_SUPPORT = True
except ImportError:
    COLOUR_SUPPORT = False

try:
    from pre_commit.store import Store

    PRE_COMMIT_AVAILABLE = True
except ImportError:
    PRE_COMMIT_AVAILABLE = False

# -- Logging -----------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# -- Git ---------------------------------------------------------------------

# The git commands in this section is partially sourced and modified from
# https://github.com/pre-commit/pre-commit/blob/main/pre_commit/git.py
# https://github.com/pre-commit/pre-commit/blob/main/pre_commit/util.py
#
# Original Copyright (c) 2014 pre-commit dev team: Anthony Sottile, Ken Struys


# prevents errors on windows
NO_FS_MONITOR = ("-c", "core.useBuiltinFSMonitor=false")
PARTIAL_CLONE = ("-c", "extensions.partialClone=true")


def no_git_env(_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """
    Clear problematic git env vars.

    Git sometimes sets some environment variables that alter its behaviour.
    You can pass `os.environ` to this method and then pass its return value
    to `subprocess.run` as a environment.

    Parameters
    ----------
    _env : Mapping[str, str] | None, optional
        A dictionary of env vars, by default None

    Returns
    -------
    dict[str, str]
        The same dictionary but without the problematic vars
    """
    # Too many bugs dealing with environment variables and GIT:
    # https://github.com/pre-commit/pre-commit/issues/300
    # In git 2.6.3 (maybe others), git exports GIT_WORK_TREE while running
    # pre-commit hooks
    # In git 1.9.1 (maybe others), git exports GIT_DIR and GIT_INDEX_FILE
    # while running pre-commit hooks in submodules.
    # GIT_DIR: Causes git clone to clone wrong thing
    # GIT_INDEX_FILE: Causes 'error invalid object ...' during commit
    _env = _env if _env is not None else os.environ
    return {
        k: v
        for k, v in _env.items()
        if not k.startswith("GIT_")
        or k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
        or k
        in {
            "GIT_EXEC_PATH",
            "GIT_SSH",
            "GIT_SSH_COMMAND",
            "GIT_SSL_CAINFO",
            "GIT_SSL_NO_VERIFY",
            "GIT_CONFIG_COUNT",
            "GIT_HTTP_PROXY_AUTHMETHOD",
            "GIT_ALLOW_PROTOCOL",
            "GIT_ASKPASS",
        }
    }


async def cmd_output(
    *cmd: str,
    check: bool = True,
    **kwargs,
) -> tuple[int, str, str]:
    """
    Run a command asyncronously.

    Parameters
    ----------
    *cmd : str
        The command to run
    check : bool, optional
        Raise an error when the returncode isn't zero, by default True
    **kwargs
        keyword arguments to pass to subprocess.Popen

    Returns
    -------
    tuple[int, str, str]
        Returncode, stderr, stdout

    Raises
    ------
    subprocess.CalledProcessError
        The command failed.
    """
    for arg in ("stdin", "stdout", "stderr"):
        kwargs.setdefault(arg, subprocess.PIPE)

    proc = await create_subprocess_exec(*cmd, **kwargs)
    stdout_bin, stderr_bin = await proc.communicate()
    stdout = stdout_bin.decode()
    stderr = stderr_bin.decode()
    returncode = cast(int, proc.returncode)

    if returncode:
        logger.debug(f"Output from '{cmd}': {stdout}")
        logger.debug(f"Err from '{cmd}': {stderr}")
        if check:
            raise subprocess.CalledProcessError(returncode, cmd, stdout, stderr)

    return returncode, stdout, stderr


async def init_repo(path: str, remote: str) -> None:
    """
    Create a minimal repository with a given remote.

    Parameters
    ----------
    path : str
        The location of the repo to create.
    remote : str
        The url to add as `origin` remote.
    """
    if os.path.isdir(remote):
        remote = os.path.abspath(remote)

    git = ("git", *NO_FS_MONITOR)
    env = no_git_env()
    # avoid the user's template so that hooks do not recurse
    await cmd_output(*git, "init", "--template=", path, env=env)
    await cmd_output(*git, "remote", "add", "origin", remote, cwd=path, env=env)


@asynccontextmanager
async def tmp_repo(repo: str) -> AsyncGenerator[Path, Any]:
    """
    Clone a repo to a temporary directory.

    This method returns a contextmanager that removes the repository on exit.

    Parameters
    ----------
    repo : str
        The repo url to clone.

    Returns
    -------
    AsyncContextManager[Path]
        A contextmanager that returns a path to the cloned directory.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _git = ("git", *NO_FS_MONITOR, "-C", tmp)
        # init repo
        await init_repo(tmp, repo)
        await cmd_output(*_git, "config", "extensions.partialClone", "true")
        await cmd_output(*_git, "config", "fetch.recurseSubmodules", "false")

        yield Path(tmp)


@lru_cache()
async def get_tags(repo_url: str, hash: str) -> List[str]:
    """
    Retrieve a list of tags for a given commit.

    This method is cached for the same combination of repo and hash.

    Parameters
    ----------
    repo_url : str
        The URL of the repo the commit is in.
    hash : str
        A valid git commit reference.

    Returns
    -------
    List[str]
        A list of tags for referencing the given commit.
    """
    async with tmp_repo(repo_url) as repo_path:
        return await get_tags_in_repo(repo_path, hash)


@lru_cache()
async def get_tags_in_repo(repo_path: str, hash: str, fetch: bool = True) -> List[str]:
    """
    Retrieve a list of tags for a given commit.

    This method is cached for the same combination of repo and hash.

    Parameters
    ----------
    repo_path : str
        The path to the cloned repo.
    hash : str
        A valid git commit reference.
    fetch : bool, optional
        Download the revision with git fetch first, by default True

    Returns
    -------
    List[str]
        A list of tags for referencing the given commit.
    """
    _git = ("git", *NO_FS_MONITOR, "-C", repo_path)

    if fetch:
        # download rev
        # The --filter options makes use of git's partial clone feature.
        # It only fetches the commit history but not the commit contents.
        # Still it fetches all commits reachable from the given commit which is way more than we need
        await cmd_output(*_git, "config", "extensions.partialClone", "true")
        await cmd_output(
            *_git, "fetch", "origin", hash, "--quiet", "--filter=tree:0", "--tags"
        )

    # determine closest tag
    closest_tag = (await cmd_output(*_git, "describe", hash, "--abbrev=0", "--tags"))[1]
    closest_tag = closest_tag.strip()

    # determine tags
    out = (await cmd_output(*_git, "tag", "--points-at", f"refs/tags/{closest_tag}"))[1]
    return out.splitlines()


@lru_cache()
async def get_hash(repo_url: str, rev: str) -> str:
    """
    Retrieve the hash for a given tag.

    This method is cached for the same combination of repo and rev.

    Parameters
    ----------
    repo_url : str
        The URL of the repo the commit is in.
    rev : str
        A valid git commit reference.

    Returns
    -------
    str
        The hex object name (hash) referenced.
    """
    async with tmp_repo(repo_url) as repo_path:
        return await get_hash_in_repo(repo_path, rev)


@lru_cache()
async def get_hash_in_repo(repo_path: str, rev: str, fetch=True) -> str:
    """
    Retrieve the hash for a given tag.

    This method is cached for the same combination of repo and rev.

    Parameters
    ----------
    repo_path : str
        The path to the cloned repo.
    rev : str
        A valid git commit reference.
    fetch : bool, optional
        Download the commit for the given reference first, by default True

    Returns
    -------
    str
        The hex object name (hash) referenced.
    """
    _git = ("git", *NO_FS_MONITOR, "-C", repo_path)
    if fetch:
        await cmd_output(
            *_git,
            "fetch",
            "origin",
            rev,
            "--quiet",
            "--depth=1",
            "--filter=tree:0",
            "--tags",
        )
    return (await cmd_output(*_git, "rev-parse", rev))[1].strip()


# -- Pre-commit --------------------------------------------------------------


def get_pre_commit_cache(repo_url: str, rev: str) -> Optional[str]:
    """
    Determine the cache location of a repository and rev.

    This tries to find the location in the pre-commit cache where the
    given rev of the given repository is already downloaded.

    Parameters
    ----------
    repo_url : str
        The URL of the repo.
    rev : str
        The revision of interest.

    Returns
    -------
    Optional[str]
        The path to the repository if available else None
    """
    if not PRE_COMMIT_AVAILABLE:
        return None

    store = Store()

    with store.connect() as db:
        result = db.execute(
            "SELECT path FROM repos WHERE repo = ? AND ref = ?",
            (repo_url, rev),
        ).fetchone()
        return result[0] if result else None


# -- Linter ------------------------------------------------------------------

# For the following code
# Copyright (c) 2023 real-yfprojects (github user)
# applies.

# sha-1 hashes consist of 160bit == 20 bytes
SHA1_LENGTH = 160

#: Minimum length of a abbrev. hex commit name (hash)
MIN_HASH_LENGTH = 7

# hex object names can also be abbreviated
regex_abbrev_hash = r"[\dabcdef]+"
pattern_abbrev_hash = re.compile(regex_abbrev_hash)


def is_hash(rev: str) -> bool:
    """
    Determine whether a revision is frozen.

    By definition a frozen revision is a complete hex object name. That is
    a SHA-1 hash. A shortened hex object name shouldn't be considered frozen
    since it might become ambiguous. Although you use SHA-1 hashes as tag names,
    the commits will take precedence over equally named tags.

    Parameters
    ----------
    rev : str
        The revision to check.

    Returns
    -------
    bool
        Whether the given revision can be considered frozen.
    """
    return bool(pattern_abbrev_hash.fullmatch(rev.lower()))


# A version can be frozen to every valid tag name or even any revision identifier
# as returned by `git describe`. The frozen comment can be followed by
# arbitrary text e.g. containing further explanation
regex_frozen_comment = r"#( frozen: (?P<rev>\S+))?(?P<note> .*)?"
pattern_frozen_comment = re.compile(regex_frozen_comment)

comment_template = "frozen: {rev}{note}"


def process_frozen_comment(comment: str) -> Optional[Tuple[str, str]]:
    """
    Check whether a comment specifies a frozen rev and extract its info.

    Parameters
    ----------
    comment : str
        The comment to process.

    Returns
    -------
    Optional[Tuple[str, str]]
        the revision and the arbitrary comment else None
    """
    match = pattern_frozen_comment.fullmatch(comment)
    if not match:
        return None

    return match["rev"], match["note"]


@enum.unique
class Rule(enum.Enum):
    """
    A enum of all enforcable rules.

    Attributes
    ----------
    code : str
        The one letter code assigned to the rule. (Must be unique)
    template : str
        A message template for complains derived from this rule
    """

    #: Issued when a revision is not frozen although required
    FORCE_FREEZE = ("f", "Unfrozen revision: {rev}")

    #: Issued when a shortened hash is used although forbidden
    NO_ABBREV = ("a", "A abbreviated hash is specified for rev: {rev}")

    #: Issued when a revision is frozen although forbidden
    FORCE_UNFREEZE = ("u", "Frozen revision: {rev}")

    #: Issued when a revision is frozen but there is no comment stating the matching tag/revision.
    MISSING_FROZEN_COMMENT = ("m", "Missing comment specifying frozen version")

    #: Issued when there is a comment of form `frozen: xxx` although the rev isn't frozen
    EXCESS_FROZEN_COMMENT = (
        "e",
        "Although rev isn't frozen the comment says so: {comment}",
    )

    #: Issued when a revision is frozen and the comment doesn't mention the tag
    #: corresponding to the given commit hash
    CHECK_COMMENTED_TAG = ("t", "Tag doesn't match frozen rev: {frozen}")

    def __new__(cls, code, template):
        """Enum constructor processing additional fields."""
        obj = object.__new__(cls)
        obj._value_ = code
        obj.code = code  # type: ignore
        obj.template = template  # type: ignore
        return obj


EXCLUSIVE_RULES = [(Rule.FORCE_FREEZE, Rule.FORCE_UNFREEZE)]


@dataclass()
class Complaint:
    """A complain derived from a rule."""

    file: str
    line: int  # starting with 0
    column: int  # starting with 0
    type: Rule
    message: str
    fixable: bool
    fixed: bool = False


class Linter:
    """Lint files and issue complains."""

    def __init__(self, rules, fix) -> None:
        """Init."""
        self.rules = rules
        self.fix = fix
        self._complains: Dict[str, List[Complaint]] = {}
        self._current_file: Optional[str] = None
        self._current_complains: Optional[List[Complaint]] = None

    @classmethod
    async def get_tags(cls, repo_url: str, rev: str) -> List[str]:
        """
        Retrieve a list of tags for a given commit.

        Parameters
        ----------
        repo_url : str
            The URL of the repo the commit is in.
        rev : str
            A valid git commit reference.

        Returns
        -------
        List[str]
            A list of tags for referencing the given commit.
        """
        logger.debug(f"Retrieving tags for {repo_url}@{rev}")
        tags = None

        try:
            cached_repo = get_pre_commit_cache(repo_url, rev)
            if cached_repo:
                logger.info(f"Found repo cached by pre-commit at {cached_repo}")
                tags = await get_tags_in_repo(cached_repo, rev)
            else:
                logger.info("Couldn't find cached repo in pre-commit cache.")
        except (sqlite3.Error, sqlite3.Warning, subprocess.CalledProcessError):
            logger.exception("Couldn't use pre-commit cache.")

        if tags is None:
            logger.debug("Checking out repo.")
            try:
                tags = await get_tags(repo_url, rev)
            except subprocess.CalledProcessError:
                logger.exception("Couldn't retrieve tags.")

        logger.debug(f"Retrieved {tags}")
        return tags or []

    @classmethod
    async def select_best_tag(cls, repo_url: str, rev: str) -> Optional[str]:
        """
        Select the best tag describing a revision.

        Parameters
        ----------
        repo_url : str
            The repo url.
        rev : str
            The commit to select a tag for.

        Returns
        -------
        Optional[str]
            The tag if any are found else None
        """
        tags = await cls.get_tags(repo_url, rev)
        tag = min(
            filter(lambda s: "." in s, tags),
            key=len,
            default=None,
        ) or min(tags, key=len, default=None)
        logger.debug(f"Selected {tag}")
        return tag

    @classmethod
    async def get_hash_for(cls, repo_url: str, rev: str) -> Optional[str]:
        """
        Retrieve the hash for a given tag.

        This method is cached for the same combination of repo and rev.

        Parameters
        ----------
        repo_url : str
            The URL of the repo the commit is in.
        rev : str
            A valid git commit reference.

        Returns
        -------
        str
            The hex object name (hash) referenced.
        """
        logger.debug(f"Retrieving hash for {repo_url}@{rev}")
        try:
            cached_repo = get_pre_commit_cache(repo_url, rev)
            if cached_repo:
                logger.info(f"Found repo cached by pre-commit at {cached_repo}")
                try:
                    return await get_hash_in_repo(cached_repo, rev, fetch=False)
                except subprocess.CalledProcessError:
                    logger.debug("Fetching tags to pre-commit cache.")
                    return await get_hash_in_repo(cached_repo, rev, fetch=True)
            else:
                logger.info("Couldn't find cached repo in pre-commit cache.")
        except (sqlite3.Error, sqlite3.Warning, subprocess.CalledProcessError):
            logger.exception("Couldn't use pre-commit cache.")

        try:
            return await get_hash(repo_url, rev)
        except subprocess.CalledProcessError:
            logger.exception("Couldn't retrieve hash.")

        return None

    def enabled(self, complain_or_rule):
        """Whether a complain or rule is enabled."""
        if isinstance(complain_or_rule, Rule):
            return complain_or_rule.value in self.rules
        if isinstance(complain_or_rule, Complaint):
            return self.enabled(complain_or_rule.type)
        raise TypeError(f"Unsupported type {type(complain_or_rule)}")

    def should_fix(self, complain_or_rule):
        """Whether fixing is enabled for a complain or rule."""
        if isinstance(complain_or_rule, Rule):
            return self.enabled(complain_or_rule) and complain_or_rule.value in self.fix
        if isinstance(complain_or_rule, Complaint):
            return complain_or_rule.fixable and self.should_fix(complain_or_rule.type)

    def complain(
        self, file: str, type_: Rule, line: int, column: int, fixable: bool, **kwargs
    ):
        """Issue a complaint."""
        msg = type_.template.format(**kwargs)  # type: ignore
        c = Complaint(file, line, column, type_, msg, fixable)

        logger.debug(f"Issued {c}")

        if self.enabled(c):
            self._complains.setdefault(file, []).append(c)
        return c

    async def lint_repo(self, repo_yaml, file: str):
        """
        Check the entry for a given repo.

        Parameters
        ----------
        repo_yaml : ruamel.yaml object
            The parsed yaml.
        file : str
            The file to issue complaints for.
        """
        complain = partial(self.complain, file)

        repo_url = repo_yaml["repo"]
        rev = repo_yaml["rev"]
        line, column = repo_yaml.lc.value("rev")

        # parse comment
        comment_rev, comment_note = None, ""
        comments = repo_yaml.ca.items.get("rev") or [None] * 3
        comment_yaml = comments[2]
        if comment_yaml:
            comment_str = comment_yaml.value.strip()
            match = process_frozen_comment(comment_str)
            comment_rev, comment_note = match if match else (None, comment_yaml.value)
            logger.debug(
                f"Split comment '{comment_str}' into "
                f"rev={comment_rev!r} note={comment_note!r}"
            )
        comment_note = comment_note or ""

        # check rev
        ih = is_hash(rev)
        is_short_hash = ih and MIN_HASH_LENGTH <= len(rev) < SHA1_LENGTH / 4  # 40
        is_full_hash = len(rev) == SHA1_LENGTH / 4  # 40

        if is_short_hash:
            complain(Rule.NO_ABBREV, line, column, False, rev=rev)

        if is_short_hash or is_full_hash:
            # frozen hash
            comp = complain(Rule.FORCE_UNFREEZE, line, column, is_full_hash, rev=rev)

            if self.should_fix(comp):
                # select best tag
                tag = await self.select_best_tag(repo_url, rev)

                if tag:
                    # adjust rev
                    repo_yaml["rev"] = tag
                    comp.fixed = True
                    is_short_hash = is_full_hash = False
                else:
                    # fixing failed
                    comp.fixable = False

        if is_short_hash or is_full_hash:
            comp = None
            if comment_rev is None:
                # no frozen: xxx comment
                comp = complain(
                    Rule.MISSING_FROZEN_COMMENT,
                    line,
                    comment_yaml.column if comment_yaml else column,
                    is_full_hash,  # need full_hash to generate comment
                    rev=rev,
                )
            elif is_full_hash and self.enabled(Rule.CHECK_COMMENTED_TAG):
                # Check the version specified in comment
                # need full_hash to identify commit

                # determine tags attached to closest commit with a tag
                tags = await self.get_tags(repo_url, rev)

                if comment_rev not in tags:
                    # wrong version
                    comp = complain(
                        Rule.CHECK_COMMENTED_TAG,
                        line,
                        comment_yaml.column if comment_yaml else column,
                        True,
                        frozen=comment_rev,
                    )

            if comp and self.should_fix(comp):  # only true when fixable
                # select best tag
                tag = await self.select_best_tag(repo_url, rev)

                if tag:
                    # adjust comment
                    repo_yaml.yaml_add_eol_comment(
                        comment_template.format(rev=tag, note=comment_note), "rev"
                    )
                    comp.fixed = True
                else:
                    # fixing failed
                    comp.fixable = False

        else:
            # unfrozen version
            comp = complain(Rule.FORCE_FREEZE, line, column, True, rev=rev)

            if self.should_fix(comp):
                # get full hash
                hash = await self.get_hash_for(repo_url, rev)

                if hash:
                    # adjust rev
                    repo_yaml["rev"] = hash
                    # adjust comment
                    repo_yaml.yaml_add_eol_comment(
                        comment_template.format(rev=rev, note=comment_note), "rev"
                    )
                    comp.fixed = True
                else:
                    # fixing failed
                    comp.fixable = False

            if not self.enabled(comp):
                if comment_rev is not None:
                    # there is a frozen: xxx comment
                    comp = complain(
                        Rule.EXCESS_FROZEN_COMMENT,
                        line,
                        comment_yaml.column or column,
                        True,
                        comment=comment_str,
                    )

                    if self.should_fix(comp):
                        if comment_note:
                            # adjust comment
                            repo_yaml.yaml_add_eol_comment(comment_note, "rev")
                        else:
                            # remove comment
                            del repo_yaml.ca.items["rev"]
                        comp.fixed = True

    async def run(self, content: str, file: str) -> Tuple[str, List[Complaint]]:
        """
        Lint a file.

        Parameters
        ----------
        content : str
            The file contents to lint.
        file : str
            The file to issue complaints for.

        Returns
        -------
        Tuple[str, List[Complaint]]
            new contents, list of complaints
        """
        # Load file
        yaml = YAML()
        yaml.preserve_quotes = True
        _, ind, bsi = load_yaml_guess_indent(content)
        yaml.indent(mapping=bsi, sequence=ind, offset=bsi)
        config_yaml = yaml.load(content)

        logger.debug(f"Indentation detected: ind={ind} bsi={bsi}")

        # Lint
        await gather(
            *(self.lint_repo(repo_yaml, file) for repo_yaml in config_yaml["repos"])
        )

        stream = StringIO()
        yaml.dump(config_yaml, stream)
        return stream.getvalue(), self._complains.get(file, [])


pattern_rich_markup_tag = r"(?<!\\)\[.*?\]"
regex_rich_markup_tag = re.compile(pattern_rich_markup_tag)


def strip_rich_markup(string: str):
    """Remove the markup for the rich library from a string."""
    return regex_rich_markup_tag.sub("", string)


def get_parser():
    """Construct a parser for the tui of this script."""
    parser = argparse.ArgumentParser()

    fix_group = parser.add_mutually_exclusive_group()
    fix_group.add_argument("--fix", default="", dest="fix")
    fix_group.add_argument(
        "--fix-all",
        action="store_const",
        const="".join(r.value for r in Rule),
        dest="fix",
    )

    rule_group = parser.add_mutually_exclusive_group()
    rule_group.add_argument("--rules", default="", dest="rules")
    rule_group.add_argument(
        "--strict",
        action="store_const",
        const="".join(r.value for r in Rule if not r == Rule.FORCE_UNFREEZE),
        dest="rules",
    )

    parser.add_argument(
        "--print",
        action="store_true",
        help="Print fixed file contents to stdout instead of writing them back into the files.",
    )
    parser.add_argument("--quiet", action="store_true", help="Don't output anything")
    parser.add_argument(
        "--format",
        help="The output format for complains. Use python string formatting. "
        "Rich markup is also supported.",
        default="{fix}[{code}] {file}:{line}:{column} {msg}",
    )

    parser.add_argument(
        "--no-colour",
        action="store_const",
        const=False,
        default=True,
        dest="colour",
        help="Disable colourful output",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)

    parser.add_argument("files", type=Path, nargs="+", metavar="file")

    return parser


@contextmanager
def output(*, colour: bool):
    """Rich output context."""
    if not COLOUR_SUPPORT:
        console = None

        def out(string: str):
            print(strip_rich_markup(string))  # noqa: T201

    else:
        console = Console(no_color=None if colour else True)

        def out(string: str):
            console.print(string, highlight=False)

    yield out, console


async def main():
    """The main entry point."""
    parser = get_parser()
    options = parser.parse_args()

    logger.setLevel(max(logging.ERROR - options.verbose * 10, 10))

    rules: str = options.rules
    output_template: str = options.format

    with output(colour=options.colour) as (out, console):
        if console:
            logger.addHandler(RichHandler(console=console))
        else:
            logger.info("Colour mode not available because rich isn't available.")

        fixed_str = "[green]FIXED[/green]"
        fixable_str = "[red]FIXABLE[/red]"
        error_str = "[red]ERROR[/red]"

        # handle exclusive rules
        for exclusive_group in EXCLUSIVE_RULES:
            found = None
            for rule in exclusive_group:
                if rule.value in rules:
                    if found:
                        parser.error(
                            f"Mutually exclusive rules `{found+rule.value}` specified"
                        )
                    found = rule.value

        linter = Linter(rules, options.fix)

        futures = []
        files: List[Path] = []
        for file in options.files:
            file = cast(Path, file)
            logger.info(f"Processing {file}...")

            content = file.read_text()

            files.append(file)
            futures.append(linter.run(content, file=file))

        results: list[Tuple[str, List[Complaint]]] = await gather(*futures)

        for file, (content, complaints) in zip(files, results):
            if options.print:
                out(content)
            elif options.fix:
                file.write_text(content)

            for comp in complaints:
                fix_status = (
                    fixed_str
                    if comp.fixed
                    else fixable_str
                    if comp.fixable
                    else error_str
                )
                out(
                    output_template.format(
                        file=comp.file,
                        code=comp.type.code,  # type: ignore[attr-defined]
                        msg=comp.message,
                        line=comp.line + 1,
                        column=comp.column + 1,
                        fix=fix_status,
                    )
                )


# TODO YAML error
# TODO Missing fields
# TODO disable flag


def run():
    """Run the program from synchronous context."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
    run()
