"""
Name harmonisation toolkit for Step 2 (applicant names in PATSTAT).

Ladder of methods, each one a function  name -> group key  that you can plug
into `evaluate()` and compare on the gold standard:

    M0  identity                       (baseline)
    M1  prof's method A                (legal-form dictionary)
    M2  cleanco                        (optional, needs `pip install cleanco`)
    M3  normalise + dictionary         (dots, accents, &, brackets, c/o ...)
    M4  M3 + fuzzy grouping            (blocking + similarity + union-find)
    M5  brand key                      (first word of the cleaned name; groups subsidiaries under the brand)
    M5b brand key + typo merge         (keys that differ by a typo are merged)
    M5+ any method + manual aliases    (parent/subsidiary fixes for the top names)
    M6  Terrorizer-style               (token/cosine matching score -> graph -> Louvain communities)
    M7  Terrorizer (official output)   (load the clusters produced by the authors' own code)

Needs pandas, numpy, scikit-learn, networkx (>=2.8 for Louvain). Optional: sentence-transformers.
"""
import re
import unicodedata
from collections import defaultdict
from math import comb
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. Cleaning
# --------------------------------------------------------------------------
# Professor's set, extended. Only removed at the START or END of a name.
LEGAL_FORMS = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "LLC", "LLP", "LP",
    "PLC", "GMBH", "AG", "KG", "MBH", "UG", "OHG", "SE", "SA", "SAS", "SARL", "SPA", "SRL", "BV", "NV",
    "AB", "AS", "ASA", "APS", "OY", "OYJ", "KK", "KABUSHIKI", "KAISHA", "LTDA", "PTY", "THE",
    "AKTIENGESELLSCHAFT", "GESELLSCHAFT", "AKTIEBOLAG", "SOCIETE", "ANONYME", "CIE", "PTE", "BHD",
    "SDN", "PVT", "LTEE", "NAAMLOZE", "VENNOOTSCHAP", "UND", "AND", "ET",
}


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalise(name):
    """Upper-case, ASCII, no punctuation, no brackets, no 'c/o ...', '&' -> AND, dots removed."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = strip_accents(str(name)).upper()
    s = re.sub(r"\bC/O\b.*$", " ", s)                 # 'NOKIA MOBILE DEVELOPMENTS LTD, C/O CLYDE' -> drop address part
    s = re.sub(r"\([^)]*\)", " ", s)                  # '(DEUTSCHLAND)', '(U.K.)'
    s = re.sub(r"\bA/S\b", " AS ", s)                 # Danish A/S before '/' is lost
    s = s.replace("&", " AND ")
    s = s.replace(".", "")                            # S.P.A. -> SPA, N.V. -> NV, INC. -> INC
    s = re.sub(r"[^A-Z0-9 ]", " ", s)                 # everything else -> space
    return " ".join(s.split())


def strip_legal_forms(norm_name, forms=LEGAL_FORMS):
    words = norm_name.split()
    while words and words[-1] in forms:
        words.pop()
    while words and words[0] in forms:
        words.pop(0)
    return " ".join(words)


def clean_dictionary_v2(name):
    """M3: normalise + legal-form dictionary. Falls back to the normalised name if nothing is left
    (e.g. a name that is only 'SA' would otherwise become '' and all such names would merge)."""
    n = normalise(name)
    c = strip_legal_forms(n)
    return c if c else n


def clean_cleanco_v2(name):
    """M2': cleanco first, then M3. Optional dependency."""
    try:
        from cleanco import basename
    except ImportError as e:                           # pragma: no cover
        raise ImportError("pip install cleanco") from e
    return clean_dictionary_v2(basename(str(name)))


# --------------------------------------------------------------------------
# 2. Fuzzy grouping (a light-weight cousin of Terrorizer: similarity graph + connected components)
# --------------------------------------------------------------------------
class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _token_sort(s):
    return " ".join(sorted(s.split()))


def similarity(a, b):
    """0..1; word-order insensitive."""
    return SequenceMatcher(None, _token_sort(a), _token_sort(b)).ratio()


def block_key(cleaned, n_chars=5):
    """Only names sharing the first `n_chars` characters are ever compared: keeps it O(n * block size)."""
    return cleaned[:n_chars]


def fuzzy_groups(cleaned_names, threshold=0.90, n_chars=5, min_len=4):
    """Return {cleaned_name: group representative}. The representative is the most 'central' (shortest) name.
    Names shorter than `min_len` are never fuzzily merged (too risky: 'IBM' vs 'IBN')."""
    names = sorted({n for n in cleaned_names if n})
    uf = UnionFind(names)
    blocks = defaultdict(list)
    for n in names:
        blocks[block_key(n, n_chars)].append(n)
    for members in blocks.values():
        for i, a in enumerate(members):
            if len(a) < min_len:
                continue
            for b in members[i + 1:]:
                if len(b) >= min_len and similarity(a, b) >= threshold:
                    uf.union(a, b)
    comp = defaultdict(list)
    for n in names:
        comp[uf.find(n)].append(n)
    rep = {}
    for members in comp.values():
        r = min(members, key=lambda x: (len(x), x))
        for m in members:
            rep[m] = r
    return rep


# --------------------------------------------------------------------------
# 3. Manual alias table (parent / subsidiary), applied on CLEANED names
# --------------------------------------------------------------------------
def load_aliases(path):
    """CSV with two columns: alias, parent  (both can be raw names; they are cleaned here)."""
    df = pd.read_csv(path)
    return {clean_dictionary_v2(a): clean_dictionary_v2(p) for a, p in zip(df["alias"], df["parent"])}


def apply_aliases(cleaned, aliases):
    return aliases.get(cleaned, cleaned)


def build_harmoniser(all_names, threshold=0.90, aliases=None, use_fuzzy=True, cleaner=clean_dictionary_v2):
    """Fit on the full list of names that will be harmonised; return a dict {raw name: final applicant}."""
    aliases = aliases or {}
    cleaned = {n: cleaner(n) for n in set(all_names)}
    rep = fuzzy_groups(cleaned.values(), threshold=threshold) if use_fuzzy else {}
    out = {}
    for raw, c in cleaned.items():
        c = rep.get(c, c)
        out[raw] = apply_aliases(c, aliases)
    return out


# --------------------------------------------------------------------------
# 3a. Brand key: the first word of the cleaned name  (the logic behind the course gold standard)
# --------------------------------------------------------------------------
# If the name STARTS with one of these words, the first word alone says nothing about who the applicant is
# ("UNIV TEXAS" vs "TEXAS INSTRUMENTS"), so the key is the first TWO words ("UNIV TEXAS").
INSTITUTION_WORDS = {
    "UNIV", "UNIVERSITY", "UNIVERSITAT", "UNIVERSITE", "UNIVERSIDAD", "UNIVERSITA", "INST", "INSTITUTE", "INSTITUT", "COLLEGE",
    "SCHOOL", "ACADEMY", "NATIONAL", "NAT", "CENTRE", "CENTER", "CENTRO", "FOUNDATION", "STATE", "THE", "REGENTS", "BOARD",
    "TRUSTEES", "CHINESE", "RESEARCH", "RES", "MAX", "FRAUNHOFER", "GESELLSCHAFT", "COMMISSARIAT", "CENTRE",
}


# Words that are too common to be a brand on their own: 'INTERNATIONAL BUSINESS MACHINES' vs 'INTERNATIONAL PAPER'.
BRAND_NEEDS_SECOND_WORD = {
    "INTERNATIONAL", "GENERAL", "AMERICAN", "UNITED", "BRITISH", "ADVANCED", "APPLIED", "FIRST", "NEW", "NORTH", "SOUTH",
    "EAST", "WEST", "GREAT", "ROYAL", "FEDERAL", "PUBLIC", "ALLIED", "CONSOLIDATED", "STANDARD", "UNIVERSAL", "NEXT",
}

# Leading words that only mean "company" / "factory" in another language: skipped, the NEXT word is the brand.
SKIP_FIRST_WORDS = {
    "COMPAGNIE", "COMPANIE", "COMPANIA", "CIE", "KONINKLIJKE", "SOCIETA", "SOCIETE", "SOCIEDAD", "FABBRICA", "FABRIQUE",
    "MANUFACTURE", "MANUFACTURA", "VEREINIGTE", "NOUVELLE", "GENERALE",
}


UNIV_WORDS = {"UNIV", "UNIVERSITY", "UNIVERSITAT", "UNIVERSITE", "UNIVERSITA", "UNIVERSIDAD", "UNIVERSITET", "UNIVERSITEIT"}
FILLER_WORDS = {"OF", "THE", "FOR", "DE", "DES", "DU", "DI", "DEL", "DELLA", "LA", "LE", "VAN", "VON", "AND", "AT", "IN"}
# words that do not identify a university by themselves: 'UNIV NORTH CAROLINA', 'NATIONAL UNIVERSITY OF SINGAPORE'
UNIV_NEEDS_MORE = BRAND_NEEDS_SECOND_WORD | {"NATIONAL", "STATE", "CITY", "CATHOLIC", "TECHNICAL", "SAN", "SAINT", "ST", "LOS",
                                              "LAS", "EL", "SAO", "MOUNT", "CHINESE", "OPEN", "ISLAMIC"}
_OWNER = re.compile(r"^(BOARD OF )?(REGENTS|TRUSTEES|GOVERNORS|CURATORS|PRESIDENT AND FELLOWS)( AND [A-Z]+)? OF (THE )?")


def _univ_key(words):
    """Universities, however they are written, get the key 'UNIV <name>':
    'UNIVERSITY OF CALIFORNIA BERKELEY', 'UNIV CALIFORNIA', 'REGENTS OF THE UNIVERSITY OF CALIFORNIA' -> 'UNIV CALIFORNIA';
    'TSINGHUA UNIVERSITY' -> 'UNIV TSINGHUA'. Returns None if the name has no university word."""
    idx = [i for i, w in enumerate(words) if w in UNIV_WORDS]
    if not idx:
        return None
    i = idx[0]
    after = [w for w in words[i + 1:] if w not in FILLER_WORDS and w not in {"SYSTEM", "SYSTEMS"}]
    before = [w for w in words[:i] if w not in FILLER_WORDS]
    if i == 0:
        if not after:
            return None
        n = 2 if after[0] in UNIV_NEEDS_MORE and len(after) > 1 else 1
        return "UNIV " + " ".join(after[:n])
    if not before:
        return None
    if all(w in UNIV_NEEDS_MORE for w in before) and after:        # NATIONAL UNIVERSITY OF SINGAPORE
        return "UNIV " + " ".join(before + after[:1])
    return "UNIV " + " ".join(before)


def brand_key(name, extra=()):
    """First word of the dictionary-cleaned name (the brand), with safeguards:
      * universities get 'UNIV <name>' whatever the word order (see _univ_key);
      * leading 'company'-type words in other languages are skipped (COMPAGNIE X -> X);
      * if the name starts with an institution word, a very common word (INTERNATIONAL, GENERAL ...), a country / city,
        a one-letter word ('D WAVE') or one of `extra` (sector words such as QUANTUM: 'QUANTUM CIRCUITS' must not merge
        with 'QUANTUM MACHINES'), the key is the first TWO words;
      * one-word names are their own key.
    Very coarse on purpose: it groups subsidiaries under their brand. Joint ventures (HITACHI MITSUBISHI ...), acronyms
    (IBM vs INTERNATIONAL BUSINESS) and brands that really are two words are its known failures: fix them with the manual alias table."""
    c = clean_dictionary_v2(name)
    c = _OWNER.sub("", c)
    words = c.split()
    while len(words) > 1 and words[0] in SKIP_FIRST_WORDS:
        words.pop(0)
    if not words:
        return ""
    uk = _univ_key(words)
    if uk:
        return uk
    prefix_words = INSTITUTION_WORDS | BRAND_NEEDS_SECOND_WORD | GEO_TOKENS | {str(w).upper() for w in extra}
    first = words[0]
    too_short = len(first) <= 2 and not any(ch.isdigit() for ch in first)      # 'D WAVE': one letter is no brand ('3M' is fine)
    if (first in prefix_words or too_short) and len(words) > 1:
        return " ".join(words[:2])
    return first


def brand_fuzzy_map(names, extra=(), threshold=0.90):
    """Brand key, then keys that differ by a typo are merged (DAIMLERCHRYSLER / DAIMLERCRYSLER).
    Safeguards: keys of 8+ letters only, no digits in the key, lengths within 2 characters, same first 4 letters,
    and the most frequent spelling becomes the label. No prefix rule (EXXON -> EXXONMOBIL): it would merge NOKIAN into NOKIA."""
    keys = {n: brand_key(n, extra) for n in names}
    count = defaultdict(int)
    for k in keys.values():
        count[k] += 1
    uniq = sorted(k for k in count if len(k) >= 8 and not any(ch.isdigit() for ch in k))
    uf = UnionFind(uniq)
    blocks = defaultdict(list)
    for k in uniq:
        blocks[k[:4]].append(k)
    for members in blocks.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if abs(len(a) - len(b)) <= 2 and similarity(a, b) >= threshold:
                    uf.union(a, b)
    comp = defaultdict(list)
    for k in uniq:
        comp[uf.find(k)].append(k)
    rep = {m: max(ms, key=lambda x: (count[x], -len(x))) for ms in comp.values() for m in ms}
    return {n: rep.get(k, k) for n, k in keys.items()}


# --------------------------------------------------------------------------
# 3b. Terrorizer-style harmonisation
#     Re-implementation of the matching + filtering stages described in
#     Ascione & Sterzi, "Presenting Terrorizer" (arXiv 2403.12083):
#       parsing   : lower-case / punctuation / legal forms removed (our clean_dictionary_v2)
#       matching  : score(i,j) = w1*[any token in common]
#                              + w2*[first token in common, if any token in common]
#                              + w3*cosine(name_i, name_j)
#                              + w4*[same web domain]            (only if you supply `augment`)
#       filtering : graph with an edge when score >= threshold, Louvain communities,
#                   each community named after its most frequent / most central name.
#     NOT included: the DuckDuckGo "knowledge augmentation" crawler, the bridgeness-centrality
#     edge pruning and the location bonus. If you run the authors' original code, load its
#     output with `load_external_clusters` instead (method M7).
# --------------------------------------------------------------------------
TERR_DEFAULTS = dict(
    w_any_token=1.0, w_first_token=1.0, w_cosine=1.0, w_domain=1.0,
    threshold=2.0,       # paper: 3.9 with 5 components incl. web info; without web info the max is 3.0
    resolution=1.0,      # Louvain resolution (paper default 1)
    max_block=400,       # tokens present in more names than this do not create candidate pairs
    extra_generic=(),    # your own generic words, e.g. ("QUANTUM", "COMPUTING") in a quantum-computing sample
    generic_min_df=50,   # data-driven generic words: in at least this many names AND in >= generic_frac of all names
    generic_frac=0.01,
)

# Words that say what KIND of organisation it is, not WHICH one. Sharing them must not count as a match
# ("UNIV CALIFORNIA" vs "UNIV TEXAS"). The paper does the same with its list of most frequent words.
GENERIC_TOKENS = {
    "UNIV", "UNIVERSITY", "UNIVERSITAT", "UNIVERSITE", "INST", "INSTITUTE", "INSTITUT", "RES", "RESEARCH", "TECH",
    "TECHNOLOGY", "TECHNOLOGIES", "SYSTEMS", "SYSTEM", "LAB", "LABS", "LABORATORY", "LABORATORIES", "NATIONAL", "NAT",
    "CENTER", "CENTRE", "SCI", "SCIENCE", "SCIENCES", "ENG", "ENGINEERING", "GROUP", "HOLDING", "HOLDINGS",
    "INTERNATIONAL", "SOLUTIONS", "DEV", "DEVELOPMENT", "SERVICES", "ELECTRONICS", "INDUSTRIES", "INDUSTRY",
    "MEDICAL", "PHARMA", "ENERGY", "DATA", "COMPUTER", "COMPUTERS", "INFORMATION", "COMMUNICATIONS", "SOFTWARE",
    "DE", "DER", "DI", "DU", "DES", "LA", "LE", "OF", "FOR", "STATE", "COLLEGE", "SCHOOL", "ACADEMY", "FOUNDATION",
}
# Geography: "SONY DEUTSCHLAND" and "PHILIPS DEUTSCHLAND" must not match because of the country. Used by the Terrorizer-style
# token match only (NOT by the brand key: "JAPAN DISPLAY" or "CHINA PETROLEUM" start with a country on purpose).
GEO_TOKENS = {
    "DEUTSCHLAND", "GERMANY", "GERMAN", "JAPAN", "AMERICA", "AMERICAN", "USA", "US", "UK", "EUROPE", "EUROPA", "EUROPEAN",
    "FRANCE", "FRANCAISE", "ITALIA", "ITALY", "ITALIANA", "CANADA", "CHINA", "KOREA", "INDIA", "SVENSKA", "SVERIGE", "SWEDEN",
    "NEDERLAND", "NETHERLANDS", "BELGIUM", "SWITZERLAND", "SUISSE", "SCHWEIZ", "AUSTRIA", "OSTERREICH", "AUSTRALIA", "ASIA",
    "PACIFIC", "GLOBAL", "WORLDWIDE", "SPAIN", "ESPANA", "ESPANOLA", "BRASIL", "BRAZIL", "MEXICO", "RUSSIA", "TAIWAN",
    "BEIJING", "SHANGHAI", "SHENZHEN", "GUANGDONG", "GUANGZHOU", "HANGZHOU", "NANJING", "WUHAN", "CHENGDU", "TIANJIN",
    "ZHEJIANG", "JIANGSU", "SICHUAN", "SHANDONG", "HEFEI", "XIAN", "SINGAPORE", "ISRAEL", "FINLAND", "DENMARK", "NORWAY", "IRELAND", "POLAND", "BELGIQUE", "LONDON", "TOKYO", "BERLIN", "PARIS",
}


def _tokens(name):
    return name.split()


def _name_matrix(cleaned_list, embed="tfidf", sbert_model="all-MiniLM-L6-v2"):
    """L2-normalised matrix, one row per cleaned name, so that cosine = row dot row.
    embed='tfidf' : char 3-4-gram TF-IDF (offline, fast, good for spelling variants)
    embed='sbert' : paper's version: SentenceTransformer token vectors, weighted by idf, averaged
    embed=callable: your own function list[str] -> 2-D array / sparse matrix
    """
    if callable(embed):
        X = embed(cleaned_list)
    elif embed == "sbert":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(sbert_model)
        toks = [_tokens(n) for n in cleaned_list]
        vocab = sorted({t for ts in toks for t in ts})
        vec = dict(zip(vocab, model.encode(vocab, normalize_embeddings=True, show_progress_bar=False)))
        n = len(cleaned_list)
        df = defaultdict(int)
        for ts in toks:
            for t in set(ts):
                df[t] += 1
        idf = {t: np.log(n / df[t]) for t in vocab}
        mx = max(idf.values()) or 1.0
        idf = {t: max(v / mx, 1e-3) for t, v in idf.items()}          # rescaled into (0, 1]
        dim = len(next(iter(vec.values())))
        X = np.zeros((n, dim))
        for i, ts in enumerate(toks):
            if ts:
                X[i] = np.mean([vec[t] * idf[t] for t in ts], axis=0)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True).fit_transform(cleaned_list)
    from sklearn.preprocessing import normalize
    return normalize(X)


def _pair_cosine(X, I, J, chunk=200_000):
    out = np.empty(len(I))
    for s in range(0, len(I), chunk):
        a, b = X[I[s:s + chunk]], X[J[s:s + chunk]]
        prod = a.multiply(b) if hasattr(a, "multiply") else a * b
        out[s:s + chunk] = np.asarray(prod.sum(axis=1)).ravel()
    return out


def _generic_set(U, p):
    "Fixed list + your own words + words that are in a large share of all names."
    n = len(U)
    df = defaultdict(int)
    for name in U:
        for t in set(_tokens(name)):
            df[t] += 1
    cut = max(p["generic_min_df"], p["generic_frac"] * n)
    return GENERIC_TOKENS | GEO_TOKENS | {str(t).upper() for t in p["extra_generic"]} | {t for t, c in df.items() if c >= cut}


def terrorizer_features(cleaned_list, embed="tfidf", max_block=400, augment=None, sbert_model="all-MiniLM-L6-v2",
                        params=None, extra_generic=()):
    """Candidate pairs + the matching-score components. Computed ONCE, then reused for tuning.
    Token features ignore generic words (see GENERIC_TOKENS); the cosine uses the whole name.
    A name made only of generic words gets no token match at all (the paper's 'type 2' names).
    augment: optional DataFrame[cleaned_name, domain] with the web domain of each name (generic domains already removed)."""
    U = list(cleaned_list)
    p = {**TERR_DEFAULTS, **(params or {})}
    if extra_generic:
        p["extra_generic"] = tuple(p["extra_generic"]) + tuple(extra_generic)
    generic = _generic_set(U, p)
    ordered = [[t for t in _tokens(n) if t not in generic] for n in U]
    toks = [set(o) for o in ordered]
    first = [(o[0] if o else "") for o in ordered]
    index = defaultdict(list)
    for i, ts in enumerate(toks):
        for t in ts:
            index[t].append(i)
    pairs = set()
    for t, ids in index.items():
        if 1 < len(ids) <= max_block:
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    pairs.add((ids[a], ids[b]))
    dom = None
    if augment is not None:
        dmap = dict(zip(augment["cleaned_name"], augment["domain"]))
        dom = [dmap.get(n) for n in U]
        by_dom = defaultdict(list)
        for i, d in enumerate(dom):
            if d:
                by_dom[d].append(i)
        for ids in by_dom.values():
            if len(ids) <= max_block:
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        pairs.add((ids[a], ids[b]))
    if not pairs:
        return U, pd.DataFrame(columns=["i", "j", "any_token", "first_token", "cosine", "domain"]), None
    I = np.fromiter((q[0] for q in pairs), dtype=np.int64, count=len(pairs))
    J = np.fromiter((q[1] for q in pairs), dtype=np.int64, count=len(pairs))
    X = _name_matrix(U, embed=embed, sbert_model=sbert_model)
    f = pd.DataFrame({
        "i": I, "j": J,
        "any_token": [1.0 if toks[a] & toks[b] else 0.0 for a, b in zip(I, J)],
        "first_token": [1.0 if (first[a] and first[a] == first[b]) else 0.0 for a, b in zip(I, J)],
        "cosine": _pair_cosine(X, I, J),
        "domain": [1.0 if (dom and dom[a] and dom[a] == dom[b]) else 0.0 for a, b in zip(I, J)],
    })
    return U, f, X


def terrorizer_communities(n_nodes, feats, params=None):
    """Edges where score >= threshold, then Louvain. Returns a list: community id per node."""
    import networkx as nx
    p = {**TERR_DEFAULTS, **(params or {})}
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    if len(feats):
        score = (p["w_any_token"] * feats["any_token"] + p["w_first_token"] * feats["first_token"]
                 + p["w_cosine"] * feats["cosine"] + p["w_domain"] * feats["domain"])
        keep = score >= p["threshold"]
        G.add_weighted_edges_from(zip(feats.loc[keep, "i"], feats.loc[keep, "j"], score[keep]))
    comms = nx.community.louvain_communities(G, weight="weight", resolution=p["resolution"], seed=42)
    label = [0] * n_nodes
    for cid, members in enumerate(comms):
        for m in members:
            label[m] = cid
    return label


def _name_communities(U, label, counts=None, X=None, naming="frequent"):
    """Pick a readable name for each community."""
    members = defaultdict(list)
    for i, c in enumerate(label):
        members[c].append(i)
    rep = {}
    for c, ids in members.items():
        if naming == "central" and X is not None and len(ids) > 2:
            sub = X[ids]
            sim = np.asarray((sub @ sub.T).mean(axis=1)).ravel() if hasattr(sub, "T") else None
            best = ids[int(np.argmax(sim))]
        else:   # most frequent, ties -> shortest
            best = max(ids, key=lambda i: ((counts or {}).get(U[i], 0), -len(U[i])))
        for i in ids:
            rep[i] = U[best]
    return rep


def terrorizer_harmonise(names, counts=None, params=None, embed="tfidf", augment=None,
                         naming="frequent", sbert_model="all-MiniLM-L6-v2"):
    """names: iterable of raw names. counts: optional {raw name: n families} used to name communities.
    Returns {raw name: community name}."""
    names = list(dict.fromkeys(names))
    cleaned = {n: clean_dictionary_v2(n) for n in names}
    cc = defaultdict(int)
    for raw, c in cleaned.items():
        cc[c] += (counts or {}).get(raw, 1)
    U = sorted({c for c in cleaned.values() if c})
    if not U:
        return {n: n for n in names}
    p = {**TERR_DEFAULTS, **(params or {})}
    U, feats, X = terrorizer_features(U, embed=embed, max_block=p["max_block"], augment=augment,
                                      sbert_model=sbert_model, params=p)
    label = terrorizer_communities(len(U), feats, p)
    if X is None and naming == "central":
        naming = "frequent"
    rep = _name_communities(U, label, cc, X, naming)
    pos = {u: i for i, u in enumerate(U)}
    return {raw: (rep[pos[c]] if c else normalise(raw) or str(raw)) for raw, c in cleaned.items()}


def tune_terrorizer(gold, n_trials=150, seed=0, embed="tfidf", test_size=0.5, max_block=400, extra_generic=()):
    """Random search (the paper uses Optuna/TPE; this has no extra dependency) maximising pairwise F1.
    The gold file is split BY GROUP into train/test so the reported test F1 is honest.
    Returns (best_params, DataFrame with train_F1/test_F1 of defaults and of the tuned parameters)."""
    rng = np.random.default_rng(seed)
    groups = gold["gold_group"].unique()
    rng.shuffle(groups)
    cut = int(len(groups) * (1 - test_size))
    tr = gold[gold["gold_group"].isin(groups[:cut])].reset_index(drop=True)
    te = gold[gold["gold_group"].isin(groups[cut:])].reset_index(drop=True)

    def prep(df):
        cl = df["name"].map(clean_dictionary_v2)
        U = sorted({c for c in cl if c})
        U, feats, X = terrorizer_features(U, embed=embed, max_block=max_block, extra_generic=extra_generic)
        return cl, U, feats

    def f1(df, prepared, params):
        cl, U, feats = prepared
        label = terrorizer_communities(len(U), feats, params)
        pos = dict(zip(U, label))
        pred = [pos.get(c, f"_empty_{i}") for i, c in enumerate(cl)]
        return precision_recall_f1(df["gold_group"], pred)["F1"]

    P_tr, P_te = prep(tr), prep(te)
    start = {**TERR_DEFAULTS, "extra_generic": tuple(extra_generic)}
    best, best_f1 = dict(start), f1(tr, P_tr, start)
    for _ in range(n_trials):
        cand = dict(
            w_any_token=rng.uniform(0.1, 1.8), w_first_token=rng.uniform(0.1, 1.8), w_cosine=rng.uniform(0.1, 1.8),
            w_domain=1.0, threshold=rng.uniform(0.5, 3.0), resolution=float(np.exp(rng.uniform(np.log(0.02), np.log(2.0)))), max_block=max_block,
            extra_generic=tuple(extra_generic))
        v = f1(tr, P_tr, cand)
        if v > best_f1:
            best, best_f1 = cand, v
    res = pd.DataFrame({
        "defaults": [f1(tr, P_tr, start), f1(te, P_te, start)],
        "tuned": [best_f1, f1(te, P_te, best)],
    }, index=["train F1", "test F1"]).round(3)
    return best, res


def load_external_clusters(path, name_col, cluster_col):
    """Official Terrorizer (or any other tool) output: a CSV/parquet with the raw name and a cluster id/label.
    Returns {raw name: cluster label}. Check the column names against the file the authors' code writes."""
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    return dict(zip(df[name_col].astype(str), df[cluster_col].astype(str)))


# --------------------------------------------------------------------------
# 3c. One entry point for Step 2
# --------------------------------------------------------------------------
METHODS = ("identity", "dictionary", "dictionary_v2", "cleanco", "fuzzy", "brand", "brand_fuzzy", "terrorizer", "external")


def harmonise_names(names, method="dictionary_v2", counts=None, aliases=None, params=None,
                    threshold=0.90, embed="tfidf", external_map=None, naming="frequent", augment=None):
    """Return {raw name: applicant label}. `aliases` (cleaned parent map) is applied last, for every method."""
    names = list(dict.fromkeys(["" if n is None or (isinstance(n, float) and pd.isna(n)) else n for n in names]))
    if method == "identity":
        out = {n: n for n in names}
    elif method == "dictionary":                       # professor's method A, as in notebook 11
        out = {n: " ".join(strip_legal_forms(" ".join(re.sub(r"[^A-Z0-9 ]", " ", str(n).upper()).split())).split()) or str(n)
               for n in names}
    elif method == "dictionary_v2":
        out = {n: clean_dictionary_v2(n) for n in names}
    elif method == "cleanco":
        out = {n: clean_cleanco_v2(n) for n in names}
    elif method == "fuzzy":
        out = build_harmoniser(names, threshold=threshold, use_fuzzy=True)
    elif method == "brand":
        extra = tuple((params or {}).get("extra_generic", ()))
        out = {n: brand_key(n, extra) for n in names}
    elif method == "brand_fuzzy":
        out = brand_fuzzy_map(names, tuple((params or {}).get("extra_generic", ())))
    elif method == "terrorizer":
        out = terrorizer_harmonise(names, counts=counts, params=params, embed=embed, naming=naming, augment=augment)
    elif method == "external":
        if external_map is None:
            raise ValueError("method='external' needs external_map=load_external_clusters(...)")
        out = {n: external_map.get(str(n), clean_dictionary_v2(n)) for n in names}     # unseen names: fall back to M3
    else:
        raise ValueError(f"method must be one of {METHODS}")
    if aliases and method != "identity":
        # 1st: the full cleaned RAW name (merge or split single names); 2nd: the resulting label (merge whole groups)
        out = {n: aliases.get(clean_dictionary_v2(n), aliases.get(clean_dictionary_v2(v), v)) for n, v in out.items()}
    return out


def gold_methods(embed="tfidf", params=None, external_map=None, aliases=None):
    """Ready-made {label: callable} for evaluate(gold, gold_methods())."""
    def wrap(m, **kw):
        def run(ns):
            ns = list(ns)
            mapping = harmonise_names(ns, method=m, aliases=aliases, **kw)      # computed ONCE for the whole list
            return [mapping[n] for n in ns]
        return run
    d = {
        "M0 identity": wrap("identity"),
        "M1 dictionary (prof A)": wrap("dictionary"),
        "M3 normalise + dictionary": wrap("dictionary_v2"),
        "M4 + fuzzy 0.90": wrap("fuzzy", threshold=0.90),
        "M5 brand key (first word)": wrap("brand", params=params),
        "M5b brand key + typo merge": wrap("brand_fuzzy", params=params),
        "M6 Terrorizer-style": wrap("terrorizer", params=params, embed=embed),
    }
    try:
        import cleanco  # noqa: F401
        d["M2 cleanco + M3"] = wrap("cleanco")
    except ImportError:
        pass
    if external_map:
        d["M7 Terrorizer (official)"] = wrap("external", external_map=external_map)
    return d


# --------------------------------------------------------------------------
# 4. Evaluation against the gold standard
# --------------------------------------------------------------------------
def _pairs(sizes):
    return sum(comb(int(n), 2) for n in sizes)


def precision_recall_f1(gold_groups, predicted_groups):
    df = pd.DataFrame({"gold": list(gold_groups), "pred": list(predicted_groups)})
    tp = _pairs(df.groupby(["gold", "pred"]).size())
    tp_fp = _pairs(df.groupby("pred").size())
    tp_fn = _pairs(df.groupby("gold").size())
    p = tp / tp_fp if tp_fp else 1.0
    r = tp / tp_fn if tp_fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"groups": df["pred"].nunique(), "precision": p, "recall": r, "F1": f1}


def evaluate(gold, methods):
    """gold: DataFrame[name, gold_group]. methods: {label: callable(list_of_names) -> list_of_group_keys}."""
    rows = {}
    for label, fn in methods.items():
        pred = fn(gold["name"].tolist())
        res = precision_recall_f1(gold["gold_group"], pred)
        try:
            from sklearn.metrics import homogeneity_completeness_v_measure as hcv
            h, c, v = hcv(gold["gold_group"], pd.factorize(pd.Series(pred))[0])
            res.update({"homogeneity": h, "completeness": c, "V": v})
        except ImportError:
            pass
        rows[label] = res
    out = pd.DataFrame(rows).T
    out["true groups"] = gold["gold_group"].nunique()
    return out.round(3)


def errors(gold, pred, kind="fp", n=20):
    """One row per mistake, so you can read them.
    kind='fp' (over-merging): predicted groups that contain several gold organisations.
        columns: predicted group | names | gold orgs mixed | example names (one per gold organisation)
    kind='fn' (under-merging): gold organisations spread over several predicted groups.
        columns: gold group | names | predicted groups | share in the main one | main group | other groups (names)"""
    df = gold.assign(pred=list(pred))
    rows = []
    if kind == "fp":
        for key, sub in df.groupby("pred"):
            k = sub["gold_group"].nunique()
            if k > 1:
                ex = sub.groupby("gold_group")["name"].first().head(4).tolist()
                rows.append({"predicted group": key, "names": len(sub), "gold orgs mixed": k, "example names": " | ".join(ex)})
        out = pd.DataFrame(rows)
        return out.sort_values(["gold orgs mixed", "names"], ascending=False).head(n).reset_index(drop=True) if len(out) else out
    for gid, sub in df.groupby("gold_group"):
        vc = sub["pred"].value_counts()
        if len(vc) > 1:
            rows.append({"gold group": gid, "example": sub["name"].iloc[0], "names": len(sub), "predicted groups": len(vc),
                         "share in main group": round(vc.iloc[0] / len(sub), 2), "main group": vc.index[0],
                         "outside main": len(sub) - vc.iloc[0],
                         "other groups (names)": ", ".join(f"{k} ({v})" for k, v in vc.iloc[1:4].items())})
    out = pd.DataFrame(rows)
    return out.sort_values("outside main", ascending=False).head(n).reset_index(drop=True) if len(out) else out


# --------------------------------------------------------------------------
# 5. Paths and saving (one place, no surprises)
# --------------------------------------------------------------------------
def find_step0_file(search_roots, patterns=("*sector_ids*", "*families*", "*qc_*")):
    """List candidate Step 0 exports under the given folders, newest first. Run it, read the list, pick one."""
    from pathlib import Path
    hits = []
    for root in search_roots:
        for pat in patterns:
            hits += [p for p in Path(root).rglob(pat) if p.suffix.lower() in (".csv", ".parquet")]
    hits = sorted(set(hits), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits
