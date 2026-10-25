"""Materialize the exact paper inputs from pinned public benchmark sources.

The repository stores only selection IDs and hashes. This command downloads
input prompts, checks their content, and writes them to an ignored data folder.
It never downloads model outputs or evaluation results.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def download(repository, revision, path):
    slug=repository.removeprefix('https://github.com/')
    if not slug or '..' in path.split('/') or '/' in revision or len(revision)!=40:
        raise ValueError('Invalid pinned source specification')
    url=f'https://raw.githubusercontent.com/{slug}/{revision}/{path}'
    request=urllib.request.Request(url,headers={'User-Agent':'selective-abstraction-input-preparation'})
    with urllib.request.urlopen(request,timeout=30) as response:
        return response.read().decode('utf-8')

def prepare(manifest, output_root, factscore_entities):
    fs=manifest['factscore']
    text=factscore_entities.read_text(encoding='utf-8')
    entities=[line.strip() for line in text.splitlines() if line.strip()]
    normalized='\n'.join(entities)+'\n'
    if len(entities)!=fs['count'] or sha(normalized)!=fs['normalized_sha256']:
        raise ValueError('FactScore input list differs from the recorded paper cohort')
    lf=manifest['longfact']
    source_by_subject={}
    records=[]
    for selected in lf['selection']:
        subject=selected['subject']
        if '/' in subject or '\\' in subject or '..' in subject:
            raise ValueError('Invalid subject identifier')
        if subject not in source_by_subject:
            raw=download(lf['upstream_repository'],lf['upstream_revision'],
                         lf['upstream_path']+'/longfact-objects_'+subject+'.jsonl')
            source_by_subject[subject]=[json.loads(line) for line in raw.splitlines() if line.strip()]
        prompt=source_by_subject[subject][selected['index']]['prompt']
        if sha(prompt)!=selected['prompt_sha256']:
            raise ValueError('LongFact prompt mismatch for '+selected['id'])
        records.append({'prompt':prompt,'subject':subject,'index':selected['index'],'id':selected['id']})
    counts=collections.Counter(x['subject'] for x in records)
    if len(records)!=76 or len(counts)!=38 or set(counts.values())!={2}:
        raise ValueError('Expected exactly two prompts from each of 38 LongFact subjects')
    files={output_root/'factscore_labeled/entities.txt':normalized,
           output_root/'longfact_objects_sampled/sampled_prompts_2.json':json.dumps(records,ensure_ascii=False,indent=2)+'\n'}
    for target,content in files.items():
        if target.exists() and target.read_text(encoding='utf-8')!=content:
            raise FileExistsError(f'Refusing to replace different existing inputs: {target}')
    for target,content in files.items():
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(content,encoding='utf-8',newline='\n')
    return {'factscore':len(entities),'longfact':len(records),'subjects':len(counts)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'configs/input_manifest.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'datasets')
    parser.add_argument('--factscore-entities',type=Path,required=True,
                        help='Official labeled FActScore prompt_entities.txt (183 entities)')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    print(json.dumps(prepare(manifest,args.output_dir,args.factscore_entities),indent=2))

if __name__=='__main__':
    main()
