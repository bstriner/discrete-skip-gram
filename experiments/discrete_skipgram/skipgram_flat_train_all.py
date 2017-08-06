import sys
import os

sys.path.append(os.path.dirname(__file__))
import skipgram_flat_els_train
import skipgram_flat_el_train
import skipgram_flat_b_train
import skipgram_flat_bw_train
import skipgram_flat_train
import skipgram_flat_l1_train
import skipgram_flat_l2_train
import skipgram_baseline
import skipgram_baseline_l1
import skipgram_baseline_l2

#skipgram_baseline.main()
#skipgram_baseline_l1.main()
#skipgram_baseline_l2.main()
skipgram_flat_train.main()
skipgram_flat_b_train.main()
skipgram_flat_el_train.main()
skipgram_flat_els_train.main()
skipgram_flat_l1_train.main()
skipgram_flat_l2_train.main()
skipgram_flat_bw_train.main()
