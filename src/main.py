import argparse

def main():
    parser = argparse.ArgumentParser(description='My Package CLI')
    parser.add_argument('data', type=str, help='Input data file')
    parser.add_argument('--option', type=str, default='default', help='Option for the algorithm')

    args = parser.parse_args()

    print(f"Processing {args.data} with option {args.option}")

if __name__ == '__main__':
    main()
