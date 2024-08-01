import argparse
import pandas as pd


# TODO implement more analysis
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    table = pd.read_csv(args.path)
    print(args.path)
    print(table)


if __name__ == "__main__":
    main()
