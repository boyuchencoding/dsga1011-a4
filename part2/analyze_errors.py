'''Error analysis for Q6: categorise wrong dev-set predictions and print counts + examples.

Usage: python analyze_errors.py results/t5_ft_experiment_dev.sql records/t5_ft_experiment_dev.pkl
'''
import re
import sys
import pickle

from load_data import load_lines

pred_sql_path, pred_rec_path = sys.argv[1], sys.argv[2]
nl = load_lines('data/dev.nl')
gt = load_lines('data/dev.sql')
pred = load_lines(pred_sql_path)
with open('records/ground_truth_dev.pkl', 'rb') as f:
    gt_recs, _ = pickle.load(f)
with open(pred_rec_path, 'rb') as f:
    pred_recs, pred_errs = pickle.load(f)

def norm(q):
    return ' '.join(q.split())

def literals(q):
    '''Quoted strings and numbers used in conditions (cities, airport codes, dates, times ...).'''
    strs = re.findall(r"'[^']*'", q)
    nums = re.findall(r'(?<![\w.])\d+(?![\w.])', q)
    return sorted(strs), sorted(n for n in nums if n != '1')  # drop the "1 = 1" filler

def tables(q):
    m = re.search(r'\bFROM\b(.*?)\bWHERE\b', q)
    seg = m.group(1) if m else ''
    return sorted(set(re.findall(r'\b([a-z_]+)\s+\1_\d+', seg)))

def select_part(q):
    m = re.search(r'SELECT(.*?)FROM', q)
    return norm(m.group(1)) if m else ''

N = len(gt)
wrong = [i for i in range(N) if set(gt_recs[i]) != set(pred_recs[i])]
cats = {'sql_error': [], 'wrong_literal': [], 'wrong_select': [], 'wrong_tables': [], 'other_logic': []}
for i in wrong:
    if pred_errs[i]:
        cats['sql_error'].append(i)
    elif literals(gt[i]) != literals(pred[i]):
        cats['wrong_literal'].append(i)
    elif select_part(gt[i]) != select_part(pred[i]):
        cats['wrong_select'].append(i)
    elif tables(gt[i]) != tables(pred[i]):
        cats['wrong_tables'].append(i)
    else:
        cats['other_logic'].append(i)

print(f'Total dev examples: {N}; record mismatches: {len(wrong)}')
print(f'SQL exact match: {sum(norm(a) == norm(b) for a, b in zip(gt, pred))}/{N}')
for k, v in cats.items():
    print(f'\n=== {k}: {len(v)}/{N}')
    for i in v[:3]:
        print(f'[{i}] NL  : {nl[i]}')
        if pred_errs[i]:
            print(f'     ERR : {pred_errs[i][:150]}')
        if k == 'wrong_literal':
            print(f'     GT lits  : {literals(gt[i])}')
            print(f'     PRED lits: {literals(pred[i])}')
        elif k == 'wrong_select':
            print(f'     GT sel  : {select_part(gt[i])}')
            print(f'     PRED sel: {select_part(pred[i])}')
        elif k == 'wrong_tables':
            print(f'     GT tables  : {tables(gt[i])}')
            print(f'     PRED tables: {tables(pred[i])}')
        else:
            print(f'     GT  : {gt[i][:400]}')
            print(f'     PRED: {pred[i][:400]}')

# Extra stats
trunc = [i for i in range(N) if pred_errs[i] and pred[i].count('(') != pred[i].count(')')]
print(f'\nUnbalanced parentheses among SQL errors (likely truncated / malformed): {len(trunc)}/{N}')
empty_pred = [i for i in wrong if not pred_errs[i] and len(pred_recs[i]) == 0 and len(gt_recs[i]) > 0]
print(f'Valid query but empty result while GT non-empty: {len(empty_pred)}/{N}')
