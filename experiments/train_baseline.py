import os
import sys

sys.path.append(os.path.dirname(__file__))
import skipgram_baseline
import skipgram_baseline_l1
import skipgram_baseline_l2


def main():
    skipgram_baseline.main()
    skipgram_baseline_l1.main()
    skipgram_baseline_l2.main()


if __name__ == "__main__":
    main()
