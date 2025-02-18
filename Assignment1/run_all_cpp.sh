#!/bin/bash
# Usage: ./run_all.sh
# This script compiles and runs all .cpp and .py files in the specified order,
# except for "gemm.py". It outputs the results in a formatted table.

mkdir -p compiled

# Arrays to store file names and outputs for the table
declare -a files=("matrix_mult.py" "st_np_matmul.py" "np_matmul.py" "matrix_mult.cpp" "st_PTB1D.cpp"
            "st_PTB16.cpp" "PTB1D.cpp" "PTB4.cpp" "PTB8.cpp" "PTB16.cpp" "PTB32.cpp" "PTB64.cpp" "accelerate.cpp") 
declare -a outputs

# Loop through the files in the custom order
for file in "${files[@]}"; do
  # Skip gemm.py for Python files
  if [[ "$file" == "gemm.py" ]]; then
    echo "Skipping $file"
    continue
  fi

  ext="${file##*.}"
  echo "=== Processing $file ($ext) ==="

  if [ "$ext" = "cpp" ]; then
    base="$(basename "$file" .cpp)"
    if [ "$base" = "accelerate" ]; then
      clang++ -O3 -DACCELRATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 "$file" -framework Accelerate -o "compiled/$base"
    else
      clang++ "$file" -O3 -march=native -ffast-math -Xpreprocessor -fopenmp \
        -I/opt/homebrew/include -L/opt/homebrew/lib -lomp \
        -o "compiled/$base"
    fi

    if [ -f "compiled/$base" ]; then
      prog_output=$(./compiled/"$base")
      single_line=$(echo "$prog_output" | tr '\n' ' ')
      outputs+=("$single_line")
    else
      outputs+=("Compilation failed.")
    fi

  elif [ "$ext" = "py" ]; then
    prog_output=$(python3 "$file")
    single_line=$(echo "$prog_output" | tr '\n' ' ')
    outputs+=("$single_line")
  else
    outputs+=("Unsupported file type.")
  fi

  echo "=== Done with $file ==="
  echo
done

# Print formatted table to terminal
echo "==================== Results Summary ===================="
printf "%-25s | %s\n" "File" "Output"
echo "---------------------------------------------------------"
for i in "${!files[@]}"; do
  printf "%-25s | %s\n" "${files[$i]}" "${outputs[$i]}"
done
echo "========================================================="
