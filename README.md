<h1 align="center">Mandelbrot Set With CUDA</h1>
<h4 align="center">(OBS: This README is not up to date, will update once code is finished)</h4>

<p align="center">
    <img src="https://github.com/Deadlica/mandelbrot-cuda/blob/main/assets/images/mandelbrot.png" alt="Mandelbrot Set" width="400" height="400">
</p>


<br />

![pull](https://img.shields.io/github/issues-pr/deadlica/mandelbrot-cuda)
![issues](https://img.shields.io/github/issues/deadlica/mandelbrot-cuda)
![coverage](https://img.shields.io/codecov/c/github/deadlica/mandelbrot-cuda)
![language](https://img.shields.io/github/languages/top/deadlica/mandelbrot-cuda)

## Introduction
This repository computes the Mandelbrot set utilizing the CUDA library for the main computation. The program comes with a CLI interface for some added flexibility with using the program.


## Dependencies
The following dependencies have been tested and are required to run the project:
- Cuda Toolkit: https://developer.nvidia.com/accelerated-computing-toolkit
- SFML library: https://www.sfml-dev.org/download.php
- GNU Make 4.4.1, or newer

`Note 1: Older versions might work, however, they have not been tested.`

Once you have the necessary dependencies you can compile the project with the following Make command:
```bash
make
```

`Note 2: Will create a CMake build when I have time.`

## Usage
There are multiple ways in which this project can be run. The program supports multiple CLI arguments that affect how the set is visualized and what parts of the set is computed.

The default setting (no CLI arguments) for the code will display the Mandelbrot set zoomed out. There is also support for smoothing with the `--smooth` flag to reduce the colored bands.

<table align="center">
  <tr>
    <td>
      <figure>
        <img src="https://github.com/Deadlica/mandelbrot-cuda/blob/main/assets/images/mandelbrot.png" alt="Mandelbrot Set" width="400" height="400">
        <figcaption>Figure 1: Mandelbrot Set (Standard)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/Deadlica/mandelbrot-cuda/blob/main/assets/images/mandelbrot_smooth.png" alt="Mandelbrot Set (Smooth)" width="400" height="400">
        <figcaption>Figure 2: Mandelbrot Set (Smooth)</figcaption>
      </figure>
    </td>
  </tr>
</table>

The usage of the `--pattern` flag will set the program to zoom into some of the well known areas of the set. The following locations are available:
* flower
* julia
* seahorse
* starfish
* sun
* tendrils
* tree

Here are some examples of using `--pattern julia`, `--pattern seahorse` and `--pattern sun` all with the `--smooth` flag:

<table align="center">
  <tr>
    <td>
      <figure>
        <img src="https://github.com/Deadlica/mandelbrot-cuda/blob/main/assets/images/julia_island.png" alt="Mandelbrot Set" width="250" height="250">
        <figcaption>Figure 3: Julia Island (Smooth)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/Deadlica/mandelbrot-cuda/blob/main/assets/images/seahorse_valley.png" alt="Mandelbrot Set (Smooth)" width="250" height="250">
        <figcaption>Figure 4: Seahorse Valley (Smooth)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/Deadlica/mandelbrot-cuda/blob/main/assets/images/sun.png" alt="Mandelbrot Set (Smooth)" width="250" height="250">
        <figcaption>Figure 5: Sun (Smooth)</figcaption>
      </figure>
    </td>
  </tr>
</table>

There are more CLI arguments available which can be found with the use of the `--help` flag.
