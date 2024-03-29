# check-pre-commit-config

[![GitHub release](https://img.shields.io/github/v/release/real-yfprojects/check-pre-commit-config?include_prereleases)](https://github.com/real-yfprojects/check-pre-commit-config/releases)
[![Github License](https://img.shields.io/github/license/real-yfprojects/check-pre-commit-config-frozen?color=bd0000)](https://github.com/real-yfprojects/check-pre-commit-config-frozen/blob/master/LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Enforce rules regarding frozen revisions in `.pre-commit-config.yaml`.

<!-- prettier-ignore-start -->
```yaml
  # Enforce frozen revisions in `.pre-commit-config.yaml`
  - repo: https://github.com/real-yfprojects/check-pre-commit-config
    rev: v1.0.0-alpha3
    hooks:
      - id: check-frozen
```
<!-- prettier-ignore-end -->

We advise to put this hook before any yaml formatters (like prettier) that you have configured as well.

## Features

This pre-commit hook ensures rules regarding the use of frozen revisions and
regarding annotating them with a `frozen: xxx` comment.
Some rules can be fixed automatically by this hook.
The following rules are supported and enabled by default with the exception
of the `u` rule. Refer to the `Args` section for information on how to customize
the behaviour of the hook. Some rules are more expensive to process since they require downloading git information for the repositories specified in `.pre-commit-config.yaml`. Conveniently this hook won't run those checks if the corresponding rules are disabled.

Revisions are considered frozen when a _hex object name_ is used. That is a hash of a commit is used as a revision. Git accepts passing only the starting letters of a _hex object name_ as long as the passed _abbreviated_ hash is unambiguous. Such abbreviated hashes are considered to be frozen revisions as well.

#### Rules

| Code | Fixable            | Enforces that...                                                                                                                   |
| ---- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `y`  | ❌ | the files can be processed by the yaml parser without error.                                                                       |
| `c`  | ❌ | the files contain the pre-commit configuration elements the other checks will run on.                                              |
| `f`  | ✅ | all revisions are frozen.                                                                                                          |
| `u`  | ✅ | no revisions are frozen.                                                                                                           |
| `a`  | ❌ | no revision is used that could be an abbreviated hash (hex object name).                                                           |
| `m`  | ✅ | there is a comment of the form `frozen: xxx` for each frozen revision.                                                             |
| `e`  | ✅ | unfrozen revisions do not have a comment of the form `frozen: xxx`                                                                 |
| `t`  | ✅ | the tag mentioned in the comment of the form `frozen: xxx` matches the hash used as a revision. (only applies for frozen comments) |

The rules `u` and `f` cannot be enabled together.

## Alternatives

You can also implement rules `f`, `u`, `m`, `e` with a more lightweight and faster pygrep hook.
That is a simple check against a regular expression:

<!-- prettier-ignore-start -->
```yaml
  - repo: local
    hooks:
      - id: rev-frozen
        name: revs must be frozen
        entry: "\\brev: (?!['\"]?[a-f0-9]{40})"
        language: pygrep
        files: '\.pre-commit-config.yaml'
      - id: rev-frozen-comment
        name: frozen revs must have a corresponding comment
        entry: "\\brev: (['\"]?)[a-f0-9]{40}\\1(?!\\s*# frozen: \\S+)"
        language: pygrep
        files: '\.pre-commit-config.yaml'
      - id: rev-unfrozen
        name: revs may not be frozen
        entry: "\\brev: ['\"]?[a-f0-9]{40}"
        language: pygrep
        files: '\.pre-commit-config.yaml'
      - id: rev-unfrozen-comment
        name: unfrozen revs may not have a contradicting comment
        entry: "\\brev: (?!(['\"]?)[a-f0-9]{40}\\1).*\\s*# frozen:"
        language: pygrep
        files: '\.pre-commit-config.yaml'
```
<!-- prettier-ignore-end -->

## Versioning

In adherence to [semver](https://semver.org/) the following rules determine compatibility between different versions of this hook.

| Version | Increased when...                                                                                                                                                                                                                                          |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MAJOR   | the behaviour or the code of a rule is changed, </br> a rule is removed, <br> the default configuration of this hook is changed, <br> commandline arguments are renamed/repurposed/removed, </br> compatibility is majorly broken in some other major way. |
| MINOR   | rules are added, </br> new commandline options are introduced.                                                                                                                                                                                             |
| PATCH   | All fixes/changes that aren't handled above.                                                                                                                                                                                                               |

## Args

When passing a list of rules for an argument the codes of the rules are expected without a delimiter. E.g.: `ycfamt` which would be equivalent to passing the `--strict` flag.

```
usage: check-pre-commit-config-frozen [-h] [--rules RULES] [--disable DISABLE] [--strict] [--fix FIX | --fix-all] [--print] [--quiet] [--format FORMAT] [--no-colour] [--verbose] file [file ...]

positional arguments:
  file               The files to enforce the rules on.

optional arguments:
  -h, --help         show this help message and exit
  --rules RULES      Enable rules.
  --disable DISABLE  Disable rules overriding any other cmd argument. When passed alone, it will enable all rules not specified.
  --strict           Enable the rules `ycfamt`.
  --fix FIX          Select rules to automatically fix.
  --fix-all          Enable all fixes available.
  --print            Print fixed file contents to stdout instead of writing them back into the files.
  --quiet            Don't output anything
  --format FORMAT    The output format for complains. Use python string formatting. Rich markup is also supported.
  --no-colour        Disable colourful output
  --verbose, -v      Increase level of debug logs displayed.
```

By default `--strict --fix-all` is passed to the underlying script.

### Examples

You can customize the hooks behaviour by overriding which command line arguments
are passed to the underlying script when the hook is run. The following configuration can be used to prevent frozen revisions from being committed:

<!-- prettier-ignore-start -->
```yaml
  # Prevent use of frozen revisions in `.pre-commit-config.yaml`
  - repo: https://github.com/real-yfprojects/check-pre-commit-config
    rev: v1.0.0-alpha3
    hooks:
      - id: check-frozen
        args:
          - "--rules"
          - "ycue"
          - "--fix-all"
```
<!-- prettier-ignore-end -->

If you just want to make sure that comments match revisions and that no abbreviated hashes are used the following configuration will be suited:

<!-- prettier-ignore-start -->
```yaml
  # Check use of `frozen: xxx` comments in `.pre-commit-config.yaml`
  - repo: https://github.com/real-yfprojects/check-pre-commit-config
    rev: v1.0.0-alpha3
    hooks:
      - id: check-frozen
        args:
          - "--rules"
          - "ycamet"
          - "--fix-all"
```
<!-- prettier-ignore-end -->

## Contributing

This project is developed in an open-source, community-driven way, as a
voluntary effort in the authors’ free time.

All contributions are greatly appreciated… pull requests are welcome,
and so are bug reports and suggestions for improvement.
The value of contributions about simplifying or otherwise improving the implementation of existing features, extending and enhancing the documentation are often underestimated or forgotten. Last but not least their are some tasks listed below that I haven't worked on yet listed below.

### TODO

-   [ ] Complete tests, achieve high test coverage
-   [ ] Automation
    -   [ ] pre-commit.ci
