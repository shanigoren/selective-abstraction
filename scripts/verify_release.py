"""Verify the explicit release file list and scan for obvious private artifacts.

This is an additional check, not a guarantee that source code is secret-free.
Review changes before updating RELEASE_FILES.txt or publishing a new version.
"""
from pathlib import Path
import hashlib
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
MANIFEST='RELEASE_FILES.txt'
PATTERNS={
    'credential-shaped literal':re.compile(r'(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)'),
    'home-directory literal':re.compile(r'(?:/home|/Users)/[A-Za-z0-9_.-]+|[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+'),
    'IPv4 endpoint other than loopback':re.compile(r'https?://(?!127\.0\.0\.1)(?:[0-9]{1,3}\.){3}[0-9]{1,3}'),
}
FORBIDDEN_SUFFIXES={'.csv','.tsv','.jsonl','.ipynb','.pkl','.pickle','.sqlite','.sqlite3','.db','.log','.pdf','.png','.jpg','.zip','.tar','.gz','.7z','.exe','.dll','.so','.dylib','.whl','.pem','.key','.p12'}

def files():
    for path in ROOT.rglob('*'):
        if '.git' in path.parts or '__pycache__' in path.parts:
            continue
        if path.is_symlink():
            raise ValueError('Symlink is not permitted: '+path.relative_to(ROOT).as_posix())
        if path.is_file():
            yield path

def main():
    expected={}
    if not (ROOT/MANIFEST).is_file():
        print('Missing RELEASE_FILES.txt; release inventory cannot be verified.')
        return 1
    for line in (ROOT/MANIFEST).read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        digest,rel=line.split('  ',1)
        if not re.fullmatch(r'[0-9a-f]{64}',digest) or Path(rel).is_absolute() or '..' in Path(rel).parts or rel in expected:
            raise ValueError('Invalid release manifest entry')
        expected[rel]=digest
    actual={p.relative_to(ROOT).as_posix():p for p in files() if p.name!=MANIFEST}
    findings=[]
    for rel in sorted(set(expected)^set(actual)):
        findings.append((rel,0,'file-list mismatch'))
    for rel,path in actual.items():
        raw=path.read_bytes()
        if rel in expected and hashlib.sha256(raw).hexdigest()!=expected[rel]:
            findings.append((rel,0,'checksum mismatch'))
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.startswith('.env'):
            findings.append((rel,0,'forbidden artifact type'))
        try:
            text=raw.decode('utf-8')
        except UnicodeDecodeError:
            findings.append((rel,0,'non-text file'))
            continue
        for number,line in enumerate(text.splitlines(),1):
            for label,pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append((rel,number,label))
    for rel,number,label in findings:
        print(f'{rel}:{number}: {label}')
    print(f'Checked {len(actual)} allowlisted files; {len(findings)} findings.')
    return 1 if findings else 0

if __name__=='__main__':
    sys.exit(main())
