from utils import count_recall_at_ks
import sys

name = sys.argv[1]
col_real = sys.argv[2]
col_pred = sys.argv[3]

count_recall_at_ks(name, col_real, col_pred).to_csv('testdesc_recall.csv', index=False)