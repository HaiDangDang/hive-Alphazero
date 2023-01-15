About
=====

Hive reinforcement learning by using [Chess AlphaZero](https://arxiv.org/pdf/1712.01815.pdf/) methods.

This project is based on these main resources:
1) The development of Chess-Alpha-zero by @Zeta36: https://github.com/Zeta36/chess-alpha-zero
2) The development of Chess-Alpha-zero using pytorch by @geochri : https://github.com/geochri/AlphaZero_Chess
3) Hive board game - python verion by @dboures: https://github.com/dboures/Hive
4) DeepMind's Oct 19th publication: [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ).
5) DeepMind Chess AlphaZero: https://arxiv.org/pdf/1712.01815.pdf

Note
----
Environment
-----------

* Python 3.8.3
* pytorch


Modules
-------

### Supervised Learning

I trained the SL model with datasets from boardspace.net: http://www.boardspace.net/hive/hivegames/

### Reinforcement Learning

This AlphaGo Zero implementation consists of three workers: `self`, `opt` and `eval`.

* `self` is Self-Play to generate training data by self-play using BestModel.
* `opt` is Trainer to train model, and generate next-generation models.
* `eval` is Evaluator to evaluate whether the next-generation model is better than BestModel. If better, replace BestModel.


### Distributed Training

Now it's possible to train the model in a distributed way. The only thing needed is to use the new parameter:

* `--type distributed`: use mini config for testing, (see `src/chess_zero/configs/distributed.py`)

So, in order to contribute to the distributed team you just need to run the three workers locally like this:

```bash
python src/chess_zero/run.py self --type distributed (or python src/chess_zero/run.py sl --type distributed)
python src/chess_zero/run.py opt --type distributed
python src/chess_zero/run.py eval --type distributed
```

### GUI
* `uci` launches the Universal Chess Interface, for use in a GUI.

To set up ChessZero with a GUI, point it to `C0uci.bat` (or rename to .sh).
For example, this is screenshot of the random model using Arena's self-play feature:
![capture](https://user-images.githubusercontent.com/4205182/34057277-e9c99118-e19b-11e7-91ee-dd717f7efe9d.PNG)

Data
-----

* `data/model/model_best_*`: BestModel.
* `data/model/next_generation/*`: next-generation models.
* `data/play_data/play_*.json`: generated training data.
* `logs/main.log`: log file.

If you want to train the model from the beginning, delete the above directories.

How to use
==========

Setup
-------
### install libraries
```bash
pip install -r requirements.txt
```

If you want to use GPU, follow [these instructions](https://www.tensorflow.org/install/) to install with pip3.

Make sure Keras is using Tensorflow and you have Python 3.6.3+. Depending on your environment, you may have to run python3/pip3 instead of python/pip.


Basic Usage
------------

For training model, execute `Self-Play`, `Trainer` and `Evaluator`.

**Note**: Make sure you are running the scripts from the top-level directory of this repo, i.e. `python src/chess_zero/run.py opt`, not `python run.py opt`.


Self-Play
--------

```bash
python src/chess_zero/run.py self
```

When executed, Self-Play will start using BestModel.
If the BestModel does not exist, new random model will be created and become BestModel.

### options
* `--new`: create new BestModel
* `--type mini`: use mini config for testing, (see `src/chess_zero/configs/mini.py`)

Trainer
-------

```bash
python src/chess_zero/run.py opt
```

When executed, Training will start.
A base model will be loaded from latest saved next-generation model. If not existed, BestModel is used.
Trained model will be saved every epoch.

### options
* `--type mini`: use mini config for testing, (see `src/chess_zero/configs/mini.py`)
* `--total-step`: specify total step(mini-batch) numbers. The total step affects learning rate of training.

Evaluator
---------

```bash
python src/chess_zero/run.py eval
```

When executed, Evaluation will start.
It evaluates BestModel and the latest next-generation model by playing about 200 games.
If next-generation model wins, it becomes BestModel.

### options
* `--type mini`: use mini config for testing, (see `src/chess_zero/configs/mini.py`)


Tips and Memory
====

GPU Memory
----------

Usually the lack of memory cause warnings, not error.
If error happens, try to change `vram_frac` in `src/configs/mini.py`,

```python
self.vram_frac = 1.0
```

Smaller batch_size will reduce memory usage of `opt`.
Try to change `TrainerConfig#batch_size` in `MiniConfig`.
