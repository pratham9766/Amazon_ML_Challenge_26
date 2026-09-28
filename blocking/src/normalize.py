"""Unicode-safe, non-destructive representations for blocking."""
import re
import unicodedata

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "llp", "ltd",
    "limited", "pvt", "private", "plc", "co", "company"
}

def nfkc(text):
    return unicodedata.normalize("NFKC", text or "")

def basic(text):
    return " ".join("".join(c if c.isalnum() else " " for c in nfkc(text).casefold()).split())

def suffix_stripped(text):
    words = basic(text).split()
    while words and words[-1] in LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)

def token_sorted(text):
    return " ".join(sorted(basic(text).split()))

def compact(text):
    return "".join(c for c in nfkc(text).casefold() if c.isalnum())

def tokens(text):
    return tuple(basic(text).split())

def numeric_tokens(text):
    return tuple(re.findall(r"\d+", nfkc(text)))

def long_tokens(text, minimum=4):
    return tuple(t for t in tokens(text) if len(t) >= minimum)

def word_features(text, bigrams=True):
    words=list(tokens(text));out={"w:"+w for w in words if len(w)>=2}
    if bigrams:
        out.update("b:"+a+"_"+b for a,b in zip(words,words[1:]))
    return out

def char_fragments(text):
    """Bounded character features for streaming inverted retrieval.

    These retain word-boundary prefixes/suffixes and spaced internal trigrams. They
    approximate a char-ngram channel without generating a dense all-pairs matrix.
    """
    out=set()
    for word in tokens(text):
        if len(word)<3: continue
        for n in (3,4,5):
            if len(word)>=n:
                out.add(f"p{n}:"+word[:n]);out.add(f"s{n}:"+word[-n:])
        if len(word)>=6:
            out.update("i3:"+word[i:i+3] for i in range(1,len(word)-2,3))
    return out

def numeric_features(text):
    nums=list(numeric_tokens(text));out={"n:"+n for n in nums if len(n)>=2}
    out.update("np:"+a+"_"+b for a,b in zip(nums,nums[1:]))
    return out

def representations(name,address):
    return {
        "name_raw":name, "name_nfkc":nfkc(name), "name_basic":basic(name),
        "name_suffix_stripped":suffix_stripped(name), "name_token_sorted":token_sorted(name),
        "name_alnum_compact":compact(name), "name_tokens":tokens(name),
        "name_numeric_tokens":numeric_tokens(name), "address_raw":address,
        "address_nfkc":nfkc(address), "address_basic":basic(address),
        "address_tokens":tokens(address), "address_charclean":compact(address),
        "address_numeric_tokens":numeric_tokens(address), "address_long_tokens":long_tokens(address)
    }
