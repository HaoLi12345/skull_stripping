# Quick Tutorial

## Install package

First download the package through git, i.e.,

    git clone https://github.com/jcreinhold/intensity-normalization.git

If you are using conda, the easiest way to ensure you are up and running is to run the  `create_env.sh` script
located in the main directory. You run that as follows:

    . ./create_env.sh

If you are *not* using conda, then you can try to install the package via the setup.py script, i.e.,
inside the `intensity-normalization` directory, run the following command:

    python setup.py install

If you don't want to bother with any of this, you can create a Docker image or Singularity image via:

    docker pull jcreinhold/intensity-normalization

or 

    singularity pull docker://jcreinhold/intensity-normalization


## Fuzzy C-means-based Normalization

Once the package is installed, if you just want to do some sort of normalization and not think too much about it, a reasonable choice is Fuzzy C-means (FCM)-based
normalization. Note that FCM requires access to a T1-w image, if this is not possible then I would recommend doing either z-score or KDE normalization
for simple normalization tasks.

Note that FCM-based normalization acts on the image by calculating the white matter (WM) mean and setting that to a specified value
(the default is 1 in the code base although that is a tunable parameter). Our FCM-based normalization method requires that
a set of scans contain a T1-w image. We use the T1-w image to create a mask of the WM over which we calculate the mean and normalize 
as previously stated (see [here](https://intensity-normalization.readthedocs.io/en/latest/algorithm.html#fuzzy-c-means) for more detail).
This mask can then be used to normalize the remaining contrasts in the set of images for a specific patient assuming that the
remaining contrast images are registered to the T1-w image.

Since all the command line interfaces (CLIs) are installed along with the package, we can run `fcm-normalize`
in the terminal to normalize a T1-w image and create a WM mask by running the following command (replacing paths as necessary):

```bash
fcm-normalize -i t1_w_image_path.nii.gz -m mask_path.nii.gz -o t1_norm_path.nii.gz -v -c t1 -s
```
 
This will output the normalized T1-w image to `t1_norm_path.nii.gz` and will create a directory 
called `wm_masks` in which the WM mask will be saved. You can then input the WM mask back in to 
the program to normalize an image of a different contrast, e.g. for T2,

```bash
fcm-normalize -i t2_image_path.nii.gz -w wm_masks/wm_mask.nii.gz -o t2_norm_path.nii.gz -v -c t2
``` 
 
You can run `fcm-normalize -h` to see more options, but the above covers most of the details necessary to 
run FCM normalization on a single image.  You can also input a directory of images like this:

```bash
fcm-normalize -i t1_imgs/ -m brain_masks/ -o out/ -v -c t1
``` 
 
and it will FCM normalize all the images in the directory `t1_imgs/` so long as the number of images and brain masks 
are equal and correspond to one another and output the normalized images into the directory `out/`.

If you want to quickly inspect the normalization results on a directory (as in the last command), you can append the
`-p` flag which will create a plot of the histograms inside the brain mask of the normalized images. For the above
case, you should expect to see alignment around the intensity level of 1 (or whatever the `--norm-value` is set to). 
You can also use the `plot-hists` CLI which is also installed (see [here](https://intensity-normalization.readthedocs.io/en/latest/exec.html#plotting)
for documentation). A use case of the `plot-hists` command would be to inspect the histograms of a set of images *before* normalization
to compare with the results of normalization.


## Other Normalization Methods

The other methods not listed above are accessible via:

1) `zscore-normalize` - do z-score normalization over the brain mask
2) `ws-normalize` - WhiteStripe normalization
3) `ravel-normalize` - RAVEL normalization
4) `kde-normalize` - Kernel Density Estimate WM peak normalization
5) `hm-normalize` - Nyul & Udupa Piecewise linear histogram matching normalization
6) `gmm-normalize` - use a GMM to normalize the WM mean over the brain mask, like FCM (do not recommend using this method!)

Note that these all have approximately the same interface with the `-i`, `-m` and `-o` options, but each 
individual method *may* need some additional input. To determine if this is the case you can either view the 
[executable documentation here](https://intensity-normalization.readthedocs.io/en/latest/exec.html) or run the command line interface (CLI) with the `-h` 
or `--help` option. To get more detail about what each of these algorithms actually does
see the [algorithm documentation here](https://intensity-normalization.readthedocs.io/en/latest/algorithm.html).

## Additional Provided Routines

There a variety of other routines provided for analysis and preprocessing. The CLI names are:

1) `coregister` - coregister via a rigid and affine transformation 
2) `plot-hists` - plot the histograms of a directory of images on one figure for comparison
3) `tissue-mask` - create a tissue mask of an input image
4) `preprocess` - resample, N4-correct, and reorient the image and mask

## Final Note

While in this tutorial we discussed interfacing with the package through command line interfaces (CLIs), 
it is worth noting that the normalization routines (and other utilities) are available as importable python functions 
which you can import, e.g.,

```python
from intensity_normalization.normalize import fcm
wm_mask = fcm.find_wm_mask(img, brain_mask)
normalized = fcm.fcm_normalize(img, wm_mask)
```
