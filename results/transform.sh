#!/bin/bash
find . -type f -name "*.csv" | while IFS= read -r file; do
  echo "Processing $file"
  mlr --csv cut -f sparql,ground_truth,generated,template,hub_id,context $file > tmp && mv tmp $file
done
