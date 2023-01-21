# hive-Alphazero
About
=====
AI play as white:


![hive_3](https://user-images.githubusercontent.com/13064213/213841954-5279c61b-27b3-4776-9f15-f90287fc4f17.gif)



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
