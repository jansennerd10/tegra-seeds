# Introduction/Requirements
This repository contains the source code to build a GPU-accelerated seed counter application that will run on the Jetson TX1.
In its default configuration, the device contains both the CUDA and OpenCV libraries, so no libraries need to be installed.
However, GNU make is required and may not be installed by default; additionally, git may not be present. Both applications
can be installed directly from the Ubuntu repository.

This application can probably be built for most devices (including x86 devices) that have nVidia GPUs with at least Maxwell architecture.
The following libraries are required:
* CUDA Toolkit
* OpenCV 2.4 (note that OpenCV 3 has breaking changes that will required updates to the code).

# File Information
* imgKernels.cu: Contains CUDA kernels (GPU functions) that help calculate the number of seeds in an image, as well as the host functions
that invoke the kernels.
* imgKernels.h: Header file containing the prototypes for the host functions in imgKernels.cu.
* main.cpp: Main program file.

# Build Information
To build this application, simply `cd` to the root directory of the local copy of this repository, and run `make`.
A "build" folder will be created that will contain a "main" executable.

The program can be invoked by the command `./main <filename>` (assuming you are currently in the "build" directory).
OpenCV supports most common image types.
