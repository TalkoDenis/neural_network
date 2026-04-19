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

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.1,
                        help='Speed of learning')

    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='The size of a butch')   
    return parser.parse_args()
