import os.path
import sys


def main():
    filename = "./counter.txt"
    if os.path.isfile(filename):
        iter_file = open(filename, "rb")
        start = int.from_bytes(iter_file.read(), sys.byteorder)
    else:
        start = 0
    print("Counter content is: {}". format(start))


if __name__ == "__main__":
    main()