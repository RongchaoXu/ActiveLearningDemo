#!/bin/bash
#!/usr/bin/php

sets=$(ls -l ./datasets |awk '/^d/ {print $NF}')
for i in $sets
do
 cases=$(ls -l ./datasets/$i |awk '/^d/ {print $NF}')
 for j in $cases
 do
  python main.py --src 'datasets'/$i/$j --type 'entropy'
 done
done
