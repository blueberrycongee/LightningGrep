"""合并所有 sft_1k*.json 到一个文件"""
import json
import glob
import os

files = glob.glob('data/code_search/sft_1k*.json')
all_data = []

for f in sorted(files):
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
            print(f'{f}: {len(data)} samples')
            all_data.extend(data)
    except Exception as e:
        print(f'{f}: ERROR - {e}')

output = 'data/code_search/sft_all.json'
with open(output, 'w', encoding='utf-8') as fp:
    json.dump(all_data, fp, ensure_ascii=False, indent=2)

print(f'\n=== Total: {len(all_data)} samples ===')
print(f'Saved to: {output}')
