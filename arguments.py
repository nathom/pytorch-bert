import argparse


def params():
    parser = argparse.ArgumentParser()

    # Experiment options
    parser.add_argument(
        "--task",
        default="baseline",
        type=str,
        help="baseline is fine-tuning bert for classification;\n\
                      tune is advanced techiques to fine-tune bert;\n\
                      constast is contrastive learning method",
        choices=["baseline", "tune", "supcon"],
    )
    parser.add_argument(
        "--temperature",
        default=0.07,
        type=int,
        help="temperature parameter for contrastive loss",
    )

    # optional fine-tuning techiques parameters
    parser.add_argument(
        "--reinit_n_layers",
        default=0,
        type=int,
        help="number of layers that are reinitialized. Count from last to first.",
    )

    # Others
    parser.add_argument(
        "--input-dir",
        default="assets",
        type=str,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        type=str,
        help="Output directory where the model predictions and checkpoints are written.",
    )
    parser.add_argument(
        "--model",
        default="bert",
        type=str,
        help="The model architecture to be trained or fine-tuned.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--dataset", default="amazon", type=str, help="dataset", choices=["amazon"]
    )

    # Key settings
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Whether to ignore cache and create a new input data",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode which is exponentially faster",
    )
    parser.add_argument(
        "--do-train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do-eval", action="store_true", help="Whether to run eval on the dev set."
    )

    # Hyper-parameters for tuning
    parser.add_argument(
        "--simclr",
        action="store_true",
        help="Use simclr loss when doing supcon",
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training and evaluation.",
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Model learning rate starting point.",
    )
    parser.add_argument(
        "--hidden-dim", default=10, type=int, help="Model hidden dimension."
    )
    parser.add_argument(
        "--drop-rate", default=0.9, type=float, help="Dropout rate for model training"
    )
    parser.add_argument(
        "--embed-dim",
        default=768,
        type=int,
        help="The embedding dimension of pretrained LM.",
    )
    parser.add_argument(
        "--adam-epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--n-epochs",
        default=10,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--n-epochs-first",
        default=10,
        type=int,
        help="Total number of training epochs first.",
    )
    parser.add_argument(
        "--step-size",
        default=5,
        type=int,
        help="Number of epochs before lowering learning rate",
    )
    parser.add_argument(
        "--gamma",
        default=0.9,
        type=float,
        help="Gamma for step LR",
    )
    parser.add_argument(
        "--max-len", default=20, type=int, help="maximum sequence length to look back"
    )

    # Specify the technique type for custom.
    parser.add_argument(
        "--technique",
        default=3,
        type=int,
        help="The technique used for the custom BERT model (1-3).",
    )

    args = parser.parse_args()
    return args
