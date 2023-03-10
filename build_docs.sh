#!/bin/bash

rm -r ./docs/*
pdoc --force --html --output-dir ./docs pykefield
mv ./docs/pykefield/* ./docs/
rm -r ./docs/pykefield

echo "Docs built with pdoc3!"
