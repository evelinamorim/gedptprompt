import json, sys

data = json.load(open(sys.argv[1],"r"))
preds = data['predictions']

# Count label distribution
from collections import Counter
all_pred_labels = [l for p in preds for l in p['pred_labels']]
all_gold_labels = [l for p in preds for l in p['gold_labels']]

print('=== PREDICTED label distribution ===')
print(Counter(all_pred_labels))

print()
print('=== GOLD label distribution ===')
print(Counter(all_gold_labels))

print()
print('=== First 5 sentences ===')
for p in preds[:5]:
    print('tokens:   ', p['tokens'])
    print('gold:     ', p['gold_labels'])
    print('predicted:', p['pred_labels'])
    print('raw:', p['raw_response'][:300])
    print('---')
