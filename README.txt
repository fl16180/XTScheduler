Code usage:

Example to run NLP finetuning:
    python nlp_transfer.py -v linear --rep 0 --task cb

Example to train CNN:
    python cnn_experiment.py -d cifar100 -m resnet20 -v xts -r 0 --eval test

Assumes Pytorch and extensions (torchvision, Huggingface), numpy, scipy, matplotlib, tqdm are installed.