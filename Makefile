all: imgKernels.cu main.cpp
	mkdir -p build
	nvcc imgKernels.cu main.cpp -o build/main -g -G `pkg-config --libs --cflags opencv`
	
clean:
	rm -rf build