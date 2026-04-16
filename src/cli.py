import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Network CLI")

    parser.add_argument('--network', 
                        type=str, 
                        default='simple', 
                        help='Choose network type')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=1000, 
                        help='Number of training loops')

    return parser.parse_args()
