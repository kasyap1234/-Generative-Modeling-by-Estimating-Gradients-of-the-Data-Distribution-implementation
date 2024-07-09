# Generative Modeling by Estimating Gradients of the Data Distribution

This repository contains the official implementation for the NeurIPS 2019 paper [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) by **Yang Song** and **Stefano Ermon** from Stanford AI Lab.

-------------------------------------------------------------------------------------

We introduce a novel method for generative modeling that estimates the derivative of the log density function (Stein score) of the data distribution. This method involves perturbing the training data with Gaussian noise of progressively smaller variances. We then estimate the score function for each perturbed data distribution by training a shared neural network called the _Noise Conditional Score Network (NCSN)_ using _score matching_. Samples can be directly generated from the NCSN using _annealed Langevin dynamics_.
## Dependencies

- PyTorch
- PyYAML
- tqdm
- pillow
- tensorboardX
- seaborn

## Running Experiments

### Project Structure

`main.py` is the main entry point for all experiments. Use `python main.py --help` to see the usage description.

```bash
usage: main.py [-h] [--runner RUNNER] [--config CONFIG] [--seed SEED]
               [--run RUN] [--doc DOC] [--comment COMMENT] [--verbose VERBOSE]
               [--test] [--resume_training] [-o IMAGE_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --runner RUNNER       The runner to execute
  --config CONFIG       Path to the config file
  --seed SEED           Random seed
  --run RUN             Path for saving running related data
  --doc DOC             A string for documentation purposes
  --verbose VERBOSE     Verbose level: info | debug | warning | critical
  --test                Whether to test the model
  --resume_training     Whether to resume training
  -o IMAGE_FOLDER, --image_folder IMAGE_FOLDER
                        The directory of image outputs
```

There are four runner classes:

- `AnnealRunner`: Main runner for experiments involving NCSN and annealed Langevin dynamics.
- `BaselineRunner`: Similar to `AnnealRunner` but uses a single fixed noise variance.
- `ScoreNetRunner`: Runner for reproducing the experiment shown in Figure 1 (Middle, Right).
- `ToyRunner`: Runner for reproducing experiments shown in Figure 2 and Figure 3.

Configuration files are stored in the `configs/` directory. For example, the configuration file for `AnnealRunner` is `configs/anneal.yml`. Log files are stored in `run/logs/doc_name`, and tensorboard files are in `run/tensorboard/doc_name`, where `doc_name` is the value provided to the `--doc` option.

### Training

To train an NCSN, use the following command:

```bash
python main.py --runner AnnealRunner --config anneal.yml --doc cifar10
```

This command trains the model based on the configuration in `configs/anneal.yml`. Log files will be saved in `run/logs/cifar10`, and tensorboard logs will be in `run/tensorboard/cifar10`.

### Sampling

To generate samples and save them to the `samples` directory, use:

```bash
python main.py --runner AnnealRunner --test -o samples
```




| CIFAR-10| ![CIFAR10](assets/cifar10_large.gif) |



## References

Large parts of the code are derived from [this GitHub repo](https://github.com/ermongroup/sliced_score_matching) (the official implementation of the [sliced score matching paper](https://arxiv.org/abs/1905.07088)).

If you find the code or ideas inspiring for your research, please consider citing the following:

```bib
@inproceedings{song2019generative,
  title={Generative Modeling by Estimating Gradients of the Data Distribution},
  author={Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11895--11907},
  year={2019}
}
```
