#!/bin/bash

# Get list of git files that exist and their current time stamps
git ls-files | xargs stat -c "%n; %y" 2> /dev/null > test_file

# Loop through files and reset mod times
while IFS="; " read -r file mtime
do
    echo "$file was modified at $mtime"
done < test_file