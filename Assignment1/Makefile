# Compiler
CXX := clang++
CXXFLAGS := -O3 -march=native -ffast-math -Xpreprocessor -fopenmp -I/opt/homebrew/include -L/opt/homebrew/lib -lomp
ACCELERATE_FLAGS := -O3 -DACCELRATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -framework Accelerate

# Directories
SRC := $(wildcard *.cpp)
OBJ_DIR := compiled
BINARIES := $(patsubst %.cpp, $(OBJ_DIR)/%, $(SRC))

# Default target: compile all
all: $(BINARIES)

# Compile each .cpp file into the compiled/ directory
$(OBJ_DIR)/%: %.cpp | $(OBJ_DIR)
ifeq ($*,accelerate)
	$(CXX) $(ACCELERATE_FLAGS) $< -o $@
else
	$(CXX) $(CXXFLAGS) $< -o $@
endif

# Create compiled directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Run a specific binary
run: 
	@if [ -z "$(file)" ]; then \
		echo "Usage: make run file=<binary>"; \
		exit 1; \
	fi; \
	./compiled/$(file)

# Clean compiled files
clean:
	rm -rf $(OBJ_DIR)

.PHONY: all run clean
