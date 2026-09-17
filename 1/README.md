# Project 1

All the raw images are in the `data` folder. I downloaded meadow, monument, and mosque (.tif) from the gallery. All images, once processed, go to the `out` folder. Displacements for green and red are printed to the CLI.

Running the starter script processes all images; you can add the filepath in `data` as a CLI argument to only process that specific file (i.e. `python3 starter.py meadow.tif`). You can also run it for all images as `python3 starter.py`.

`starter.py` makes few modifications to the starter code discussed in the blog, but is largely the same. `utils.py` has all my helper functions for calculating L2 norm & NCC, as well as to compute the metric over the image (`check_image`), align the image (`align`), and put it all together recursively (`align_recursive`).
