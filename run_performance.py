import argparse

from src.model_performance import model_performance


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run performance metrics")
    parser.add_argument('--labels',
                        help='The location of the supervised labels images.',
                        default='projects/nuclear/resources/images/raw-labels-stacks-complete-cc')
    parser.add_argument('--predictions',
                        help='The location of the predictions images.',
                        default='projects/nuclear/resources/images/predictions-stacks-cc')

    args = parser.parse_args()
    labels_dir = args.labels
    predictions_dir = args.predictions

    print('\n=====> Running performance metrics')
    model_performance(labels_dir, predictions_dir)
