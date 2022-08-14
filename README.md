# Compressing multilingual machine translation models with knowledge distillation

Code for the thesis titled:
> "Compressing multilingual machine translation models with knowledge distillation"

> Alexandra Spyrou
>
>MSc Artificial Intelligence, QMUL

### Data:

To process data, see the script

```
util_scripts/prepare_multilingual_data.sh
```

### Training Scripts:

The training scripts for one-to-many (O2M) translation of the related language group is under the
directory ```job_scripts/related_ted8_o2m/```.

Our methods:

Baselines:

Proportional:

```bash 
job_scripts/related_ted8_o2m/proportional.sh 
``` 

Temperature:

```bash 
job_scripts/related_ted8_o2m/temperature.sh 
```

Latent depth (Sparse transformer):

```bash 
job_scripts/related_ted8_o2m/latent_depth.sh 
```

The scripts for Diverse O2M is under the directory ```job_scripts/diverse_ted8_o2m/```

For Knowledge Distillation:

To train the teacher and save its output distributions:

```bash 
runs/train_teacher.sh 
```

To train the student by interpolating the distillation and Negative Log-Likelihood loss

```bash 
runs/train_distill_student.sh 
```

### Inference Scripts:

Each of the experiment script directory contains a trans.sh file to translate the test set. To translate the test set
for the Related O2M Proportional:

```bash
job_scripts/related_ted8_o2m/trans.sh checkpoints/related_ted8_o2m/proportional/ 
``` 

To perform inference on the Sparse transformer you need to use the trans_latent.sh file to translate the test set.

```bash
job_scripts/related_ted8_o2m/trans_latent.sh checkpoints/related_ted8_o2m/latent_depth/ 
``` 

To translate other experiment, simply replace the argument with the experiment checkpoint directory.
