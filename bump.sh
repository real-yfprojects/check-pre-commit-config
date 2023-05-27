#!/usr/bin/env bash

set -e

new_tag=$1

# determine old tag
git fetch origin HEAD --tags --quiet
old_tag="$(git describe HEAD --abbrev=0 --tags)"

echo Bump "$old_tag" '->' "$new_tag"

# Ensure clean workspace

if ! (git update-index --refresh && git diff-index --quiet HEAD --); then
    echo Commit staged and unstaged changes first.
    exit 2
fi

# update references of old version in README
sed -i -e "s/$old_tag/$new_tag/g" README.md

# Commit
msg='Bump version from `'$old_tag'` to `'$new_tag'`'
git add README.md
git commit --no-verify -m "$msg"
