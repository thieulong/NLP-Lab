from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from config import FALLBACK_PATTERN_PREFIX
from spacy_utils import coarse_ent_type

@dataclass(frozen=True)
class CandidatePair:
    head_text: str
    tail_text: str
    head_type: str
    tail_type: str
    head_span: Tuple[int, int]
    tail_span: Tuple[int, int]
    pattern: str

def propose_pairs(sent, window_k: int = 2) -> List[CandidatePair]:
    """
    Propose entity pairs likely supported by the sentence structure.

    Patterns:
      P1  SVO:          VERB nsubj -> dobj
      P2  pred+prep:    predicate (VERB/ADJ) with nsubj/nsubjpass and prep/agent->pobj
                        (IMPORTANT: includes passive 'agent' such as "was founded by X")
      P3  copular prep: predicate NOUN/ADJ is ROOT, has 'cop' and nsubj, plus prep->pobj
      P4  poss copular: possessor + (capital/founder/etc) + cop + attr
                        also handles "capital of Russia is Moscow" (prep 'of')
      P5  apposition:   X, CEO Y (appos)
      P6  noun-of:      (role noun) + prep 'of' -> entity, connect subj/possessor -> of_obj
      P7  relcl recovery: who/which/that subject -> antecedent entity
      P8  conj expand:  if X->Y found and X has conjuncts, add conjunct->Y (also for Y)
      P0  fallback:     within-window ordered pairs between entity mentions (noisy)
                        + additional general safety: max token-distance between entity roots
    """

    # --- general fallback safety knob (not dataset-specific) ---
    MAX_FALLBACK_TOKEN_DIST = 18  # reduce if you want even safer fallback

    # token index -> entity tuple for tokens inside an entity span
    tok2ent: Dict[int, Tuple[str, str, Tuple[int, int]]] = {}
    ents = list(sent.ents)

    # span -> (ent_text, coarse_type, span) + root token index for distance gating
    span2ent: Dict[Tuple[int, int], Tuple[str, str, Tuple[int, int]]] = {}
    span2root_i: Dict[Tuple[int, int], int] = {}

    for ent in ents:
        t = coarse_ent_type(ent.label_)
        span = (ent.start_char - sent.start_char, ent.end_char - sent.start_char)
        e_tuple = (ent.text, t, span)
        span2ent[span] = e_tuple
        # root token position within the sentence (token indices are doc-level, but diff still works)
        span2root_i[span] = ent.root.i

        for tok in ent:
            tok2ent[tok.i] = e_tuple

    def ent_from_token(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        return tok2ent.get(tok.i)

    def first_ent_in_subtree(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        for t in tok.subtree:
            e = tok2ent.get(t.i)
            if e:
                return e
        return None

    def ent_or_subtree(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        return ent_from_token(tok) or first_ent_in_subtree(tok)

    def antecedent_entity_for_relpron(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        head = tok.head
        for _ in range(3):
            e = ent_or_subtree(head)
            if e and e[0].lower() not in {"who", "which", "that"}:
                return e
            head = head.head
        return None

    def add_pair(out: List[CandidatePair], h, t, pattern: str) -> None:
        h_text, h_type, h_span = h
        t_text, t_type, t_span = t
        if not h_text or not t_text or h_text == t_text:
            return
        out.append(CandidatePair(
            head_text=h_text,
            tail_text=t_text,
            head_type=h_type,
            tail_type=t_type,
            head_span=h_span,
            tail_span=t_span,
            pattern=pattern,
        ))

    def add_pair_with_conj(out: List[CandidatePair], h, t_tok, pattern: str) -> None:
        """
        Add pair h -> entity(t_tok) if present, and also h -> each conjunct entity of t_tok.
        This is robust for 'by Steve Jobs and Steve Wozniak' cases.
        """
        t_ent = ent_or_subtree(t_tok)
        if t_ent:
            add_pair(out, h, t_ent, pattern)
        # conjunct expansion at token-level (in addition to later P8)
        for conj in t_tok.conjuncts:
            t2 = ent_or_subtree(conj)
            if t2:
                add_pair(out, h, t2, pattern + "+conj")

    cands: List[CandidatePair] = []

    # P1: SVO (active voice only)
    for tok in sent:
        if tok.pos_ != "VERB":
            continue
        subj = None
        obj = None
        for ch in tok.children:
            if ch.dep_ == "nsubj":
                subj = ent_or_subtree(ch)
                if subj and subj[0].lower() in {"who", "which", "that"}:
                    subj = antecedent_entity_for_relpron(ch)
            elif ch.dep_ == "dobj":
                obj = ent_or_subtree(ch)
        if subj and obj:
            add_pair(cands, subj, obj, "P1_SVO")

    # P2: predicate with prep/agent pobj (handles passive: "was founded by X")
    for tok in sent:
        if tok.pos_ not in ("VERB", "ADJ"):
            continue

        subj = None
        for ch in tok.children:
            if ch.dep_ in ("nsubj", "nsubjpass"):
                subj = ent_or_subtree(ch)
                if subj and subj[0].lower() in {"who", "which", "that"}:
                    subj = antecedent_entity_for_relpron(ch)
        if not subj:
            continue

        for ch in tok.children:
            # IMPORTANT: passive agent is often dep_ == "agent"
            if ch.dep_ not in ("prep", "agent"):
                continue

            # Optional: if you want to be stricter, prioritize 'by' as agent
            # but keep it general by allowing all preps/agents.
            # if ch.dep_ == "agent" and ch.lemma_.lower() != "by": continue

            for grand in ch.children:
                if grand.dep_ == "pobj":
                    # Add subj -> pobj and subj -> pobj conjuncts
                    add_pair_with_conj(cands, subj, grand, "P2_pred_prep")

    # P3: copular ROOT NOUN/ADJ with prep pobj
    for tok in sent:
        has_cop = any(ch.dep_ == "cop" for ch in tok.children)
        if not has_cop:
            continue
        subj = None
        for ch in tok.children:
            if ch.dep_ == "nsubj":
                subj = ent_or_subtree(ch)
        if not subj:
            continue
        for ch in tok.children:
            if ch.dep_ != "prep":
                continue
            for grand in ch.children:
                if grand.dep_ == "pobj":
                    pobj = ent_or_subtree(grand)
                    if pobj:
                        add_pair(cands, subj, pobj, "P3_copular_prep")

    # P4: possessive copular bridging (possessor -> attr)
    for tok in sent:
        has_cop = any(ch.dep_ == "cop" for ch in tok.children)
        if not has_cop:
            continue
        possessor = None
        attr_ent = None
        for ch in tok.children:
            if ch.dep_ == "poss":
                possessor = ent_or_subtree(ch)
            elif ch.dep_ == "attr":
                attr_ent = ent_or_subtree(ch)

        # also handle "X of Y is Z"
        of_obj = None
        for ch in tok.children:
            if ch.dep_ == "prep" and ch.lemma_.lower() == "of":
                for grand in ch.children:
                    if grand.dep_ == "pobj":
                        of_obj = ent_or_subtree(grand)

        if attr_ent and possessor:
            add_pair(cands, possessor, attr_ent, "P4_poss_cop_attr")
        if attr_ent and of_obj:
            add_pair(cands, of_obj, attr_ent, "P4_of_cop_attr")

    # P5: apposition
    for tok in sent:
        if tok.dep_ != "appos":
            continue
        appos_ent = ent_or_subtree(tok)
        head_ent = ent_or_subtree(tok.head)
        if appos_ent and head_ent:
            add_pair(cands, head_ent, appos_ent, "P5_appos")
            add_pair(cands, appos_ent, head_ent, "P5_appos_rev")

    # P6: noun-of role phrases
    for tok in sent:
        if tok.pos_ != "NOUN":
            continue
        of_obj = None
        for ch in tok.children:
            if ch.dep_ == "prep" and ch.lemma_.lower() == "of":
                for grand in ch.children:
                    if grand.dep_ == "pobj":
                        of_obj = ent_or_subtree(grand)
        if not of_obj:
            continue

        possessor = None
        for ch in tok.children:
            if ch.dep_ == "poss":
                possessor = ent_or_subtree(ch)
        if possessor:
            add_pair(cands, possessor, of_obj, "P6_noun_of_poss")

        has_cop = any(ch.dep_ == "cop" for ch in tok.children)
        if has_cop:
            subj = None
            for ch in tok.children:
                if ch.dep_ == "nsubj":
                    subj = ent_or_subtree(ch)
            if subj:
                add_pair(cands, subj, of_obj, "P6_noun_of_cop_subj")

    # Dedup before conj expansion
    seen = set()
    uniq: List[CandidatePair] = []
    for c in cands:
        key = (c.head_span, c.tail_span, c.pattern)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    # P8: conjunction expansion (span-based)
    span2tokens: Dict[Tuple[int, int], List[Any]] = {}
    for tok in sent:
        e = ent_from_token(tok)
        if not e:
            continue
        span2tokens.setdefault(e[2], []).append(tok)

    def conjunct_entities(ent_span: Tuple[int, int]) -> List[Tuple[str, str, Tuple[int, int]]]:
        toks = span2tokens.get(ent_span, [])
        out = []
        for tok in toks:
            for conj in tok.conjuncts:
                e2 = ent_or_subtree(conj)
                if e2:
                    out.append(e2)
        seen_sp = set()
        uniq_out = []
        for e2 in out:
            if e2[2] in seen_sp:
                continue
            seen_sp.add(e2[2])
            uniq_out.append(e2)
        return uniq_out

    expanded: List[CandidatePair] = list(uniq)
    for c in uniq:
        for h2 in conjunct_entities(c.head_span):
            expanded.append(CandidatePair(
                head_text=h2[0], head_type=h2[1], head_span=h2[2],
                tail_text=c.tail_text, tail_type=c.tail_type, tail_span=c.tail_span,
                pattern=c.pattern + "+P8_head_conj",
            ))
        for t2 in conjunct_entities(c.tail_span):
            expanded.append(CandidatePair(
                head_text=c.head_text, head_type=c.head_type, head_span=c.head_span,
                tail_text=t2[0], tail_type=t2[1], tail_span=t2[2],
                pattern=c.pattern + "+P8_tail_conj",
            ))

    # Final dedup
    final_seen = set()
    final: List[CandidatePair] = []
    for c in expanded:
        key = (c.head_span, c.tail_span, c.pattern)
        if key in final_seen:
            continue
        final_seen.add(key)
        final.append(c)

    # P0: fallback window between nearby entity mentions (ordered pairs)
    # Safer fallback: require entities to be within MAX_FALLBACK_TOKEN_DIST by root token index
    if window_k and window_k > 0:
        existing_pairs = {(c.head_span, c.tail_span) for c in final}

        ent_items = []
        for ent in ents:
            t = coarse_ent_type(ent.label_)
            span = (ent.start_char - sent.start_char, ent.end_char - sent.start_char)
            root_i = ent.root.i
            ent_items.append((ent.text, t, span, root_i))

        # keep original order of mentions, but skip pairs that are too far apart token-wise
        for i, e1 in enumerate(ent_items):
            for j in range(i + 1, min(len(ent_items), i + 1 + window_k)):
                e2 = ent_items[j]
                dist = abs(e1[3] - e2[3])
                if dist > MAX_FALLBACK_TOKEN_DIST:
                    continue

                # forward
                if (e1[2], e2[2]) not in existing_pairs and e1[0] != e2[0]:
                    final.append(CandidatePair(
                        head_text=e1[0], head_type=e1[1], head_span=e1[2],
                        tail_text=e2[0], tail_type=e2[1], tail_span=e2[2],
                        pattern=f"{FALLBACK_PATTERN_PREFIX}",
                    ))
                    existing_pairs.add((e1[2], e2[2]))

                # reverse
                if (e2[2], e1[2]) not in existing_pairs and e1[0] != e2[0]:
                    final.append(CandidatePair(
                        head_text=e2[0], head_type=e2[1], head_span=e2[2],
                        tail_text=e1[0], tail_type=e1[1], tail_span=e1[2],
                        pattern=f"{FALLBACK_PATTERN_PREFIX}_rev",
                    ))
                    existing_pairs.add((e2[2], e1[2]))

    return final
