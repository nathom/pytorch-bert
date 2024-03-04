import torch
from torch import nn
from tqdm import tqdm as progress_bar

from arguments import params
from dataloader import (
    check_cache,
    get_dataloader,
    prepare_features,
    prepare_inputs,
    process_data,
)
from load import load_data, load_tokenizer
from model import CustomModel, IntentModel, SupConModel
from utils import check_directories, set_seed, setup_gpus

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("using MPS")

elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("using CUDA")
else:
    device = torch.device("cpu")
    print("using CPU")


def baseline_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets["train"], split="train")

    # task2: setup model's optimizer_scheduler if you have
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=args.step_size, gamma=args.gamma
    # )
    model.optimizer = optimizer
    # model.scheduler = scheduler

    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            inputs, labels = prepare_inputs(batch, model, device=device)
            # inputs = {k: v.to(device) for k, v in inputs.items()}
            # labels = labels.to(device)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            # model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()

        run_eval(args, model, datasets, tokenizer, split="validation")
        print("epoch", epoch_count, "| losses:", losses)


def custom_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader

    # task2: setup model's optimizer_scheduler if you have

    # task3: write a training loop


def run_eval(args, model, datasets, tokenizer, split="validation"):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model, device=device)
        logits = model(inputs, labels)

        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()

    print(
        f"{split} acc:",
        acc / len(datasets[split]),
        f"|dataset split {split} size:",
        len(datasets[split]),
    )


def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss

    criterion = SupConLoss(temperature=args.temperature)

    # task1: load training split of the dataset

    # task2: setup optimizer_scheduler in your model

    # task3: write a training loop for SupConLoss function


if __name__ == "__main__":
    args = params()
    args = setup_gpus(args)
    args = check_directories(args)
    set_seed(args)

    cache_results, already_exist = check_cache(args)
    tokenizer = load_tokenizer(args)

    if already_exist:
        features = cache_results
    else:
        data = load_data()
        features = prepare_features(args, data, tokenizer, cache_results)
    datasets = process_data(args, features, tokenizer)
    for k, v in datasets.items():
        print(k, "length", len(v))

    if args.task == "baseline":
        model = IntentModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split="validation")
        run_eval(args, model, datasets, tokenizer, split="test")
        baseline_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split="test")
    elif (
        args.task == "custom"
    ):  # you can have multiple custom task for different techniques
        model = CustomModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split="validation")
        run_eval(args, model, datasets, tokenizer, split="test")
        custom_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split="test")
    elif args.task == "supcon":
        model = SupConModel(args, tokenizer, target_size=60).to(device)
        supcon_train(args, model, datasets, tokenizer)
