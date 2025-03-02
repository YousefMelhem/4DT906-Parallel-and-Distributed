# Compiler
CXX := clang++
OUTDIR := compiled

# Common Flags
CXXFLAGS := -O3 -march=native -mcpu=native -ftree-vectorize -ffast-math -Xpreprocessor -fopenmp \
            -I/opt/homebrew/include -L/opt/homebrew/lib -lomp

# Accelerate Framework Flags
ACCELERATE_FLAGS := -O3 -DACCELRATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -framework Accelerate

# Source Files
SRC := $(wildcard *.cpp)
OBJS := $(SRC:.cpp=)

# Default target
all: $(patsubst %, $(OUTDIR)/%, $(OBJS))

# Compilation rules
$(OUTDIR)/accelerate: accelerate.cpp | $(OUTDIR)
	$(CXX) $(ACCELERATE_FLAGS) $< -o $@

$(OUTDIR)/%: %.cpp | $(OUTDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# Create output directory if it doesn't exist
$(OUTDIR):
	mkdir -p $(OUTDIR)

# Run the compiled program
run: all
	@for bin in $(OBJS); do \
		echo "Running $$bin..."; \
		./$(OUTDIR)/$$bin; \
	done

# Clean compiled files
clean:
	rm -rf $(OUTDIR)
