#!/bin/sh

python3 setup.py sdist bdist_wheel && git add . && git commit -am "New revision" && git push
