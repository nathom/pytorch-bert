import torch
from torch import nn
from tqdm import tqdm as progress_bar
from transformers import get_linear_schedule_with_warmup

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
from utils import check_directories, set_seed, setup_gpus, plot_stuff, compare_and_save

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

    # To hold the loss of each epoch.
    train_acc_pts, train_loss_pts = [], []
    valid_acc_pts, valid_loss_pts = [], []

    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        acc, losses = 0, 0
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

            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()

            model.optimizer.step()  # backprop to update the weights
            losses += loss.item()

        acc /= len(datasets['train'])
        train_acc_pts.append(acc)
        train_loss_pts.append(losses)

        outs = run_eval(args, model, datasets, tokenizer, split="validation")
        valid_acc_pts.append(outs[0])
        valid_loss_pts.append(outs[1])

        print("epoch:", epoch_count, "| acc:", acc, "| losses:", losses)
    
    plot_stuff(args, train_acc_pts, "Train", valid_acc_pts, "Valid", "Accuracy")
    plot_stuff(args, train_loss_pts, "Train", valid_loss_pts, "Valid", "Loss")


def custom_train(args, model, datasets, tokenizer, technique=1):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    epochs = args.n_epochs
    train_dataloader = get_dataloader(args, datasets["train"], split="train")

    # Technique 2 uses the learning rate decay (LLRD).
    if technique == 3 or technique == 2:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    model.optimizer = optimizer
    model.scheduler = scheduler

    # To hold the loss of each epoch.
    train_acc_pts, train_loss_pts = [], []
    valid_acc_pts, valid_loss_pts = [], []

    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        acc, losses = 0, 0
        model.train()

        for step, batch in progress_bar(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            inputs, labels = prepare_inputs(batch, model, device=device)

            if technique == 3 or technique == 2:
                optimizer.zero_grad()

            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights

            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()

            # Technique 1 uses the schdeuler.
            if technique == 3 or technique == 1:
                model.scheduler.step()  # Update learning rate schedule
            losses += loss.item()

        acc /= len(datasets['train'])
        train_acc_pts.append(acc)
        train_loss_pts.append(losses)

        outs = run_eval(args, model, datasets, tokenizer, split="validation")
        valid_acc_pts.append(outs[0])
        valid_loss_pts.append(outs[1])

        print("epoch:", epoch_count, "| acc:", acc, "| losses:", losses)
    
    plot_stuff(args, train_acc_pts, "Train", valid_acc_pts, "Valid", "Accuracy")
    plot_stuff(args, train_loss_pts, "Train", valid_loss_pts, "Valid", "Loss")


def run_eval(args, model, datasets, tokenizer, split="validation"):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc, losses = 0, 0
    criterion = nn.CrossEntropyLoss()
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model, device=device)
        logits = model(inputs, labels)
        if len(logits.shape) == 3:
            logits = logits[:, 0, :].squeeze(1)

        loss = criterion(logits, labels)
        losses += loss.item()

        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()

    print(
        f"{split} acc:",
        acc / len(datasets[split]),
        f"|dataset split {split} size:",
        len(datasets[split]),
    )

    # Return the accuracy and loss.
    return (acc / len(datasets[split]), losses)


def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss

    criterion = SupConLoss(temperature=args.temperature)
    epochs = args.n_epochs
    train_dataloader = get_dataloader(args, datasets["train"], split="train")

    # task2: setup model's optimizer_scheduler if you have
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    model.optimizer = optimizer
    assert args.n_epochs_first > 1
    for epoch in range(args.n_epochs_first):
        losses = 0
        model.train()

        for step, batch in progress_bar(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            inputs, labels = prepare_inputs(batch, model, device=device)
            optimizer.zero_grad()
            logits1 = model(inputs, labels)
            logits2 = model(inputs, labels)
            logits = torch.cat([logits1.unsqueeze(1), logits2.unsqueeze(1)], dim=1)
            loss = criterion.forward(logits, labels, device=device)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            # model.scheduler.step()  # Update learning rate schedule
            losses += loss.item()

        run_eval(args, model, datasets, tokenizer, split="validation")
        print("cse epoch", epoch, "| losses:", losses)

    for param in model.encoder.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    model.optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    model.scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    for epoch in range(args.n_epochs):
        losses = 0
        model.train()

        for batch in progress_bar(train_dataloader):
            inputs, labels = prepare_inputs(batch, model, device=device)
            optimizer.zero_grad()
            logits = model(inputs, labels)
            loss = criterion.forward(logits, labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            model.scheduler.step()  # Update learning rate schedule
            losses += loss.item()

        run_eval(args, model, datasets, tokenizer, split="validation")
        print("regular epoch", epoch, "| losses:", losses)

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
        org = run_eval(args, model, datasets, tokenizer, split="test")
        baseline_train(args, model, datasets, tokenizer)
        fin = run_eval(args, model, datasets, tokenizer, split="test")
        compare_and_save(args, [("before_test", org), ("after_test", fin)])
    elif (
        args.task == "tune"
    ):  # you can have multiple custom task for different techniques
        model = CustomModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split="validation")
        run_eval(args, model, datasets, tokenizer, split="test")
        custom_train(args, model, datasets, tokenizer, technique=args.technique)
        fin = run_eval(args, model, datasets, tokenizer, split="test")
        compare_and_save(args, [(f"technique_{args.technique}", fin)])
    elif args.task == "supcon":
        model = SupConModel(args, tokenizer, target_size=60).to(device)
        supcon_train(args, model, datasets, tokenizer)
