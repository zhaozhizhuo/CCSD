import argparse


def generate_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--device", type=str, default='cuda', help="device")
    parser.add_argument("--epochs", type=int, default=30, help="epochs")
    parser.add_argument("--model_path", type=str, default='./bert-base-uncased', help="model_path")

    #laat
    parser.add_argument("--d_a", type=int, default=256, help="d_a")
    parser.add_argument("--n_labels", type=int, default=148, help="n_labels")

    return parser
