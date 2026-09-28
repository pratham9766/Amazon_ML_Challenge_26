"""Conservative diagnostic forms; none modifies the official inputs."""
import re
import unicodedata

LEGAL = {'inc', 'incorporated', 'corp', 'corporation', 'llc', 'llp', 'ltd',
         'limited', 'pvt', 'private', 'plc', 'co', 'company'}

def norm(s):
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in
                            unicodedata.normalize('NFKC', s).lower()).split())

def suffix(s):
    words = norm(s).split()
    while words and words[-1] in LEGAL:
        words.pop()
    return ' '.join(words)

def numbers(s):
    return re.findall(r'\d+', s)

def postal(s):
    return re.findall(r'(?<!\w)\d{5,6}(?!\w)', s)

def forms(s):
    n = norm(s)
    return {'raw': s, 'lowercase': s.lower(),
            'unicode_nfkc': unicodedata.normalize('NFKC', s),
            'punctuation_removal': ''.join(c for c in s.lower() if not unicodedata.category(c).startswith('P')),
            'ampersand_and': norm(s.replace('&', ' and ')),
            'whitespace': ' '.join(s.split()), 'normalized': n,
            'legal_suffix': suffix(s), 'token_sort': ' '.join(sorted(n.split())),
            'digit_extraction': ' '.join(numbers(s))}
