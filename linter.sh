#!/bin/bash -ev

# Run this script at project root by "./linter.sh" before you commit.

{
	black --version | grep "19.3b0" > /dev/null
} || {
	echo "Linter requires black==19.3b0 !"
	exit 1
}

echo "Running isort..."
isort -y -sp .

echo "Running black..."
black .

echo "Running flake8..."
if [ -x "$(command -v flake8-3)" ]; then
  flake8-3 .
else
  python3 -m flake8 .
fi

command -v arc > /dev/null && {
  echo "Running arc lint ..."
  arc lint
}
