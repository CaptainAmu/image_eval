## Image Evaluation Package

This is to be used jointly with the ```diffusers``` experiment repo, in which results are stored in ```MY_EXPERIMENTS/FK_correct_loras/output```.

To run image evaluation, use the virtual environment ```image_eval``` using ```image_eval.yaml```. 

Then copy this ```output``` folder to this ```image_eval``` folder. Note this ```output``` should contain a folder ```images``` of images and ```results.csv``` with column ```image_name``` indicating the corresponding image names.

Then run 

```sbatch job.sh```

The computed CLIP score and LAION score will be attached to the original csv to form a new ```metrics_results.csv``` in the ```output``` folder.
