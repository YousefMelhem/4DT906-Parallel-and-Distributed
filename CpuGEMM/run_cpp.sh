#!/bin/bash
# Usage: ./run_cpp.sh filename.cpp

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 filename.cpp"
  exit 1
fi

FILE="$1"

if [ ! -f "$FILE" ]; then
  echo "Error: File '$FILE' not found."
  exit 1
fi



NAME=$(basename "$FILE" .cpp)

if [ "$NAME" = "accelerate" ]; then
  clang++ -O3 -DACCELRATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 "$FILE" -framework Accelerate -o compiled/"$NAME" && ./compiled/"$NAME"
else
  clang++ "$FILE" -mfma -std=c++17 -mfma -Xpreprocessor -fopenmp -O3 -march=native -mcpu=native -ffast-math  \
    -I/opt/homebrew/include -L/opt/homebrew/lib -lomp -o compiled/"$NAME" && ./compiled/"$NAME"
fi