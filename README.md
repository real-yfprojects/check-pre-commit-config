# check-pre-commit-config-frozen

[![Latest GitHub release](https://img.shields.io/github/v/release/real-yfprojects/check-pre-commit-config)](https://github.com/real-yfprojects/check-pre-commit-config/releases)
[![Github License](https://img.shields.io/github/license/real-yfprojects/check-pre-commit-config-frozen?color=bd0000)](https://github.com/real-yfprojects/check-pre-commit-config-frozen/blob/master/LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Enforce rules regarding frozen revisions in `pre-commit-config.yaml`.

```yaml
# Enforce frozen revisions in `.pre-commit-config.yaml`
- repo: https://github.com/real-yfprojects/check-pre-commit-config
  rev: v1.0.0
  hooks:
      - id: check-frozen
```

## Features

This pre-commit hook ensures rules regarding the use of frozen revisions and
regarding annotating them with a `frozen: xxx` comment.
Some rules can be fixed automatically by this hook.
The following rules are supported and enabled by default with the exception
of the `u` rule. Refer to the `Args` section for information on how to customize
the behaviour of the hook. Some rules are more expensive to process since they require downloading git information for the repositories specified in `.pre-commit-config.yaml`. Conveniently this hook won't run those checks if the corresponding rules are disabled.

Revisions are considered frozen when a _hex object name_ is used. That is a hash of a commit is used as a revision. Git accepts passing only the starting letters of a _hex object name_ as long as the passed _abbreviated_ hash is unambiguous. Such abbreviated hashes are considered to be frozen revisions as well.

#### Rules

<!-- For some reason github improperly displays the ✅ without colour -->

| Code | Fixable            | Enforces that...                                                                                                                   |
| ---- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `y`  | ❌                 | the files can be processed by the yaml parser without error.                                                                       |
| `c`  | ❌                 | the files contain the pre-commit configuration elements the other checks will run on.                                              |
| `f`  | :heavy_check_mark: | all revisions are frozen.                                                                                                          |
| `u`  | :heavy_check_mark: | no revisions are frozen.                                                                                                           |
| `a`  | ❌                 | no revision is used that could be an abbreviated hash (hex object name).                                                           |
| `m`  | :heavy_check_mark: | there is a comment of the form `frozen: xxx` for each frozen revision.                                                             |
| `e`  | :heavy_check_mark: | unfrozen revisions do not have a comment of the form `frozen: xxx`                                                                 |
| `t`  | :heavy_check_mark: | the tag mentioned in the comment of the form `frozen: xxx` matches the hash used as a revision. (only applies for frozen comments) |

The rules `u` and `f` cannot be enabled together.

## Versioning

In adherence to [semver](https://semver.org/) the following rules determine compatibility between different versions of this hook.

| Version | Increased when...                                                                                                                                                                                                                                          |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MAJOR   | the behaviour or the code of a rule is changed, </br> a rule is removed, <br> the default configuration of this hook is changed, <br> commandline arguments are renamed/repurposed/removed, </br> compatibility is majorly broken in some other major way. |
| MINOR   | rules are added, </br> new commandline options are introduced.                                                                                                                                                                                             |
| PATCH   | All fixes/changes that aren't handled above.                                                                                                                                                                                                               |

## Args

When passing a list of rules for an argument the codes of the rules are expected without a delimiter. E.g.: `ycfamt` which would be äquivalent to passing the `--strict` flag.

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

```yaml
# Prevent use of frozen revisions in `.pre-commit-config.yaml`
- repo: https://github.com/real-yfprojects/check-pre-commit-config
  rev: v1.0.0
  hooks:
      - id: check-frozen
        args:
            - "--rules"
            - "ycue"
            - "--fix-all"
```

If you just want to make sure that comments match revisions and that no abbreviated hashes are used the following configuration will be suited:

```yaml
# Check use of `frozen: xxx` comments in `.pre-commit-config.yaml`
- repo: https://github.com/real-yfprojects/check-pre-commit-config
  rev: v1.0.0
  hooks:
      - id: check-frozen
        args:
            - "--rules"
            - "ycamet"
            - "--fix-all"
```

## Contributing

This project is developed in an open-source, community-driven way, as a
voluntary effort in the authors’ free time.

All contributions are greatly appreciated… pull requests are welcome,
and so are bug reports and suggestions for improvement.
The value of contributions about simplifying or otherwise improving the implementation of existing features, extending and enhancing the documentation are often underestimated or forgotten. Last but not least their are some tasks listed below that I haven't worked on yet listed below.

### TODO

-   [ ] Write tests
-   [ ] Automation
    -   [ ] pre-commit.ci
    -   [ ] update version in README config snippet
-   [ ] Write documentation (README)
-   [ ] Register hook on pre-commit.com
