import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Network CLI")

    parser.add_argument(
        "--network", type=str, default="simple", help="Choose network type"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of training loops"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="Speed of learning"
    )

    parser.add_argument(
        "--batch_size", type=int, default=2, help="The size of a batch"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="src/data/houses.csv",
        help="Path to the dataset",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="price",
        help="The column we want to predict",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data used for the final exam",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the trained model to the folder",
    )

    return parser.parse_args()


def parse_inference_args():
    parser = argparse.ArgumentParser(description="CLI")
    # parser.add_argument('--size',
    #                     type=float,
    #                     required=True,
    #                     help='Size of the first variable')

    # parser.add_argument('--bedrooms',
    #                     type=float,
    #                     required=True,
    #                     help='Size of the second variable')

    parser.add_argument(
        "--data",
        type=str,
        default="src/data/houses.csv",
        help="Path of thr dataset",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="price",
        help="The column we want to predict",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        type=float,
        required=True,
        help="A list of input values",
    )

    return parser.parse_args()
