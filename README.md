# MMM: Exploring Conditional Multi-Track Music Generation with the Transformer and the Johann Sebastian Bach Chorales Dataset.

Implementation of the paper "MMM: Exploring Conditional Multi-Track Music Generation with the Transformer" ([paper](https://arxiv.org/abs/2008.06048)). Uses OpenAI's GPT-2 to compose music.

## Contact.

Find me on [LinkedIn](https://www.linkedin.com/in/dr-tristan-behrens-734967a2/) and say hello.

If you find and issue or have a feature request, report either here on GitHub. 

Please be so kind and star the repository if you find it useful.

## Acknowledgements.

This repository has been created in cooperation with [Pyoneer](https://www.pyoneer.io). I am very grateful!

## About.

This repository allows you to train GPT-2 on the Johann Sebastian Bach chorale dataset. You can train both MMMTrack and MMMBar from the paper.

## How to run.

Requirements:

```
pip install transformers
pip install tokenizers
pip install torch
pip install music21
pip install note_seq
```

Training:

1. Clone this repository `git clone https://github.com/AI-Guru/MMM-JSB.git`.
2. Train MMMTrack with `python train_jsb_mmmtrack.py`.
3. Train MMMBar with `python train_jsb_mmmbar.py`.

Sampling: Run the jupyter notebook.

Training should take roughly one hour on a GPU per model for the JSB dataset.

## Pretrained checkpoint.

A pretrained network can be found here:
https://ai-guru.s3.eu-central-1.amazonaws.com/mmm-jsb/mmm_jsb_checkpoints.zip

## What is missing?

- TensorFlow support is rudimentary.
- Data preprocessing and training on the Lakh dataset.
- Implementation as a tool or a DAW plugin.

## License.

Released under the Apache-2.0 License.