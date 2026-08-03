from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

from .ingest import iter_document_chunks, write_jsonl
from .morphology import extract_stems
from ..schemas import DocumentChunk, EvidenceRef


COMMON_PRODUCT = "\uacf5\ud1b5"
RRF_K = 60
# _compact()/_metadata_hints()\uac00 \ubb38\uc790\uc5f4 \uc555\ucd95\uc6a9\uc73c\ub85c\ub9cc \uc0ac\uc6a9 - \uac80\uc0c9 \ud1a0\ud070\ud654\ub294 kiwipiepy(_tokens).
TOKEN_RE = re.compile(r"[\uac00-\ud7a3A-Za-z0-9]+")
CASE_INTENT_TOKENS = {"case", "cases", "dispute", "판례", "사례", "분쟁", "조정", "청구", "보상"}
PRODUCT_INTENT_TOKENS = {"상품", "약관", "특약", "설명서"}
GUIDE_INTENT_TOKENS = {"안내", "민원", "접수", "칭찬", "불만", "처리", "회신", "홈페이지", "소비자보호", "영업일"}

# \uba85\uc0ac\ub958(NNG/NNP/NNB/NR)\uc640 \uc678\ub798\uc5b4\u00b7\uc22b\uc790(SL/SN)\ub9cc \ub0a8\uae34\ub2e4. \uc870\uc0ac(JKS/JX/...)\ub098 \uc5b4\ubbf8\ub97c
# \ud3ec\ud568\ud558\uba74 "\uc801\uae08\uc744"\uacfc "\uc801\uae08"\uc774 \ub2e4\ub978 \ud1a0\ud070\uc774 \ub418\uc5b4 \uc815\ud655 \uc77c\uce58 \uac80\uc0c9\uc774 \uae68\uc9c4\ub2e4 - \uc2e4\uce21\uc73c\ub85c
# \ud655\uc778\ub41c \uc0c1\ud488\uc124\uba85\uc11c recall \uc800\ud558\uc758 \uc8fc\ub41c \uc6d0\uc778.
_CONTENT_TAGS = frozenset({"NNG", "NNP", "NNB", "NR", "SL", "SN"})
_kiwi = None


def _kiwi_instance():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi

        _kiwi = Kiwi(num_workers=-1)
    return _kiwi


def _morphs_to_tokens(morphs) -> set[str]:
    return {morph.form.lower() for morph in morphs if morph.tag in _CONTENT_TAGS}


def _tokens(text: str) -> set[str]:
    """\uc9c8\uc758 \uc2dc\uc810 \ub4f1, \uc0ac\uc804 \uacc4\uc0b0\ub41c \ud1a0\ud070\uc774 \uc5c6\uc744 \ub54c \uc4f0\ub294 \ub2e8\uac74 \ud615\ud0dc\uc18c \ubd84\uc11d."""
    return _morphs_to_tokens(_kiwi_instance().tokenize(text))


def tokenize_many(texts: Sequence[str]) -> list[list[str]]:
    """corpus \ube4c\ub4dc \uc2dc \ubc30\uce58\ub85c \ubbf8\ub9ac \uacc4\uc0b0\ud574 \uce90\uc2f1\ud558\uae30 \uc704\ud55c \uba40\ud2f0\uc2a4\ub808\ub4dc \ud1a0\ud070\ud654.

    65,000\uac1c \uccad\ud06c\ub97c \uac74\ubcc4\ub85c tokenize()\ud558\uba74 \uc11c\ubc84 \uae30\ub3d9\ub9c8\ub2e4 3\ubd84 \ub118\uac8c \uac78\ub9b0\ub2e4
    (\uc2e4\uce21). corpus \ube4c\ub4dc \uc2dc \ud55c \ubc88\ub9cc \uacc4\uc0b0\ud574 DocumentChunk.tokens\uc5d0 \uc800\uc7a5\ud574\ub450\uba74
    SearchIndex.__init__\uc740 \uadf8 \uacb0\uacfc\ub97c \uadf8\ub300\ub85c \uc77d\uae30\ub9cc \ud574\uc11c \ube60\ub974\ub2e4.
    """
    return [sorted(_morphs_to_tokens(result)) for result in _kiwi_instance().tokenize(list(texts))]


def _compact(text: str) -> str:
    return "".join(TOKEN_RE.findall(text)).lower()


def _text_key(text: str) -> str:
    """상품설명서마다 반복되는 정형 문구는 doc_id가 달라도 같은 근거다."""
    return " ".join(text.split())


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _corpus_for(chunk: DocumentChunk) -> str:
    path = chunk.path.split(":", 1)[-1].replace("\\", "/")
    if chunk.doc_type == "glossary" or path.startswith("dictionary/"):
        return "glossary"
    if chunk.doc_type == "case" or path.startswith("cases/"):
        return "cases"
    if chunk.doc_type == "guide" or path.startswith("guides/"):
        return "guides"
    if chunk.doc_type == "law" or path.startswith(("regulations/", "공통규정/")):
        return "regulations"
    if path.startswith("products/"):
        return "products"
    return "other"


def _metadata_text(chunk: DocumentChunk) -> str:
    path = chunk.path.split(":", 1)[-1].replace("\\", "/")
    filename = Path(path).stem
    return " ".join(
        part
        for part in (filename, chunk.section or "", " ".join(chunk.product))
        if part
    )


def _metadata_hints(text: str) -> set[str]:
    hints: set[str] = set()
    suffixes = ("상품설명서", "설명서", "특약", "약관", "상품")
    for token in _tokens(text):
        compact = _compact(token)
        if len(compact) < 4:
            continue
        hints.add(compact)
        without_date = re.sub(r"\d{4,}$", "", compact)
        if len(without_date) >= 4:
            hints.add(without_date)
        for suffix in suffixes:
            if suffix in without_date:
                stem = without_date.split(suffix, 1)[0]
                if len(stem) >= 4:
                    hints.add(stem)
    return hints


def _intent(query: str, query_tokens: set[str]) -> str | None:
    compact_query = _compact(query)
    guide_signals = {"민원", "접수", "칭찬", "불만", "처리", "회신", "소비자보호", "영업일"}
    case_signals = {"판례", "사례", "분쟁", "조정", "청구", "보상"}
    product_signals = {"상품", "약관", "특약", "설명서"}
    if any(signal in compact_query for signal in guide_signals):
        return "guides"
    if any(signal in compact_query for signal in case_signals):
        return "cases"
    if any(signal in compact_query for signal in product_signals):
        return "products"
    if query_tokens & GUIDE_INTENT_TOKENS:
        return "guides"
    if query_tokens & CASE_INTENT_TOKENS:
        return "cases"
    if query_tokens & PRODUCT_INTENT_TOKENS:
        return "products"
    return None


def load_jsonl(path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    if not path.exists():
        return chunks
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                chunks.append(DocumentChunk.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError):
                continue
    return chunks


def document_manifest(data_dir: Path) -> dict[str, dict[str, int]]:
    manifest: dict[str, dict[str, int]] = {}
    paths = list(data_dir.rglob("*.pdf"))
    paths.extend((data_dir / "regulations" / "law_api").glob("*.json"))
    for path in sorted(paths):
        if path.name == "manifest.json":
            continue
        stat = path.stat()
        manifest[path.relative_to(data_dir).as_posix()] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return manifest


def changed_documents(
    current: dict[str, dict[str, int]],
    previous: dict[str, dict[str, int]],
) -> list[str]:
    return sorted(
        path
        for path in set(current) | set(previous)
        if current.get(path) != previous.get(path)
    )


def needs_reindex(data_dir: Path, chunks_path: Path) -> bool:
    if not chunks_path.exists():
        return True
    artifact_mtime = chunks_path.stat().st_mtime_ns
    sources = (
        list(data_dir.rglob("*.pdf"))
        + list((data_dir / "regulations" / "law_api").glob("*.json"))
        + [
            path
            for path in (
                data_dir / "cases" / "cases.csv",
                data_dir / "dictionary" / "fine_financial_glossary.csv",
            )
            if path.exists()
        ]
    )
    return any(
        path.stat().st_mtime_ns > artifact_mtime
        for path in sources
        if path.name != "manifest.json"
    )


def reindex(data_dir: Path, output: Path) -> int:
    return write_jsonl(iter_document_chunks(data_dir), output)


_NAME_SIGNAL_DOC_TYPES = frozenset({"product_manual", "rate_table"})


class SearchIndex:
    def __init__(self, chunks: list[DocumentChunk], *, source: str = "memory"):
        self.chunks = chunks
        self.source = source
        self._active_today: tuple[date, frozenset[int]] | None = None
        self._token_sets: list[set[str]] = []
        self._metadata_token_sets: list[set[str]] = []
        self._compact_metadata: list[str] = []
        self._metadata_hints: list[set[str]] = []
        self._compact_text: list[str] = []
        self._stem_sets: list[set[str]] = []
        self._token_index: dict[str, set[int]] = defaultdict(set)
        self._stem_index: dict[str, set[int]] = defaultdict(set)
        self._document_frequency: Counter[str] = Counter()
        # 상품설명서는 폴더 단위 카테고리(예금/적금)만으로는 "KB 스타적금3"처럼
        # 같은 카테고리의 다른 상품과 구분이 안 된다. ingest.py가 파일명에서 뽑은
        # 상품명을 section에 채워두므로, 그 상품명 자체를 토큰화해 캐싱하고
        # search()에서 별도 가중치를 준다. 상품 문서만 대상이라(전체의 1.4%) 매
        # 기동마다 즉석 형태소 분석해도 무시할 만한 비용이다.
        self._name_tokens: list[set[str] | None] = []
        for index, chunk in enumerate(chunks):
            tokens = set(chunk.tokens) if chunk.tokens is not None else _tokens(chunk.text)
            metadata = _metadata_text(chunk)
            self._token_sets.append(tokens)
            self._metadata_token_sets.append(_tokens(metadata))
            self._compact_metadata.append(_compact(metadata))
            self._metadata_hints.append(_metadata_hints(metadata))
            self._compact_text.append(_compact(chunk.text))
            # chunk.stems는 오프라인 precompute(stem_corpus.py)로 채워지며 cases/
            # products/guides 청크만 값을 갖는다 - regulations는 빈 리스트라 이
            # 보너스가 그냥 0으로 남는다 (형태소 분석기가 없거나 미실행이어도 동일).
            stem_set = set(chunk.stems)
            self._stem_sets.append(stem_set)
            for token in tokens:
                self._token_index[token].add(index)
                self._document_frequency[token] += 1
            for stem in stem_set:
                self._stem_index[stem].add(index)
            if chunk.doc_type in _NAME_SIGNAL_DOC_TYPES and chunk.section and not chunk.section.startswith("page:"):
                self._name_tokens.append(_tokens(chunk.section))
            else:
                self._name_tokens.append(None)

    def _active_indices(self, target: date) -> frozenset[int]:
        """Indices of chunks whose effective-date bounds include `target`.

        ~96% of regulation chunks are expired versions kept only so a
        historically-dated complaint can still find the regulation that was
        in force then. For the common case (as_of is today) a wide query term
        can still pull thousands of those expired chunks into the per-token
        candidate union, only to be dropped one by one in the loop below by
        the exact same date check. Intersecting with this precomputed,
        day-cached set prunes them before the loop instead of during it -
        same IDF base (self._document_frequency, len(self.chunks) - both
        computed over the full corpus, unchanged), so scores are identical to
        an unpruned scan; only which candidates get visited changes.
        """
        if self._active_today is None or self._active_today[0] != target:
            active = frozenset(
                index
                for index, chunk in enumerate(self.chunks)
                if not (chunk.effective_from and chunk.effective_from > target)
                and not (chunk.effective_to and chunk.effective_to < target)
            )
            self._active_today = (target, active)
        return self._active_today[1]

    @classmethod
    def from_jsonl(cls, path: Path, *, exclude_doc_types: frozenset[str] = frozenset()) -> "SearchIndex":
        chunks = load_jsonl(path)
        if exclude_doc_types:
            chunks = [chunk for chunk in chunks if chunk.doc_type not in exclude_doc_types]
        return cls(chunks, source=f"jsonl:{path}")

    @classmethod
    def from_data_dir(
        cls,
        data_dir: Path,
        *,
        chunks_path: Path | None = None,
        exclude_doc_types: frozenset[str] = frozenset(),
    ) -> "SearchIndex":
        if chunks_path and chunks_path.exists() and not needs_reindex(data_dir, chunks_path):
            index = cls.from_jsonl(chunks_path, exclude_doc_types=exclude_doc_types)
            if index.chunks:
                return index
        # iter_document_chunks never yields glossary/case rows - those are synthesized
        # separately by build_corpus.py - so no filtering needed on this fallback path.
        return cls(list(iter_document_chunks(data_dir)), source=f"data:{data_dir}")

    def search(
        self,
        query: str,
        *,
        product: str | None = None,
        as_of: date | None = None,
        top_k: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[EvidenceRef]:
        query_tokens = _tokens(query)
        if not query_tokens and not query_embedding:
            return []

        if query_tokens:
            candidate_indices = set().union(
                *(self._token_index.get(token, set()) for token in query_tokens)
            )
        else:
            candidate_indices = set(range(len(self.chunks)))

        compact_query = _compact(query)
        compact_terms = {term for term in (_compact(token) for token in query_tokens) if len(term) >= 4}
        if compact_query:
            for index, compact_metadata in enumerate(self._compact_metadata):
                if len(compact_metadata) >= 4 and (
                    compact_metadata in compact_query or compact_query in compact_metadata
                ):
                    candidate_indices.add(index)
                    continue
                if any(hint in compact_query for hint in self._metadata_hints[index]):
                    candidate_indices.add(index)
                    continue
                compact_text = self._compact_text[index]
                if compact_text and any(term in compact_text for term in compact_terms):
                    candidate_indices.add(index)

        # "연장했더니"(질의) vs "연장시"(문서), "금리가" vs "금리를" 처럼 조사·어미만
        # 다른 형태는 _tokens()의 정규식 매칭으로는 절대 겹치지 않는다. 어간만 뽑은
        # stems로 한 번 더 겹침을 본다 - 코퍼스 쪽은 stem_corpus.py가 미리 계산해
        # chunk.stems에 넣어둔 값이라 여기서는 질의만 실시간으로 분석하면 된다.
        query_stems = set(extract_stems(query))
        if query_stems:
            candidate_indices.update(
                set().union(*(self._stem_index.get(stem, set()) for stem in query_stems))
            )

        intent = _intent(query, query_tokens)
        if intent == "guides":
            candidate_indices.update(
                index for index, chunk in enumerate(self.chunks) if _corpus_for(chunk) == "guides"
            )

        # as_of=None disables date filtering entirely below (`if as_of and ...`
        # short-circuits), so only prune when a date was actually given -
        # pruning on a None as_of would silently start filtering candidates
        # that the per-candidate checks were never going to filter anyway.
        if as_of == date.today():
            candidate_indices &= self._active_indices(as_of)

        query_idf = {
            token: math.log1p(
                (len(self.chunks) - self._document_frequency.get(token, 0) + 0.5)
                / (self._document_frequency.get(token, 0) + 0.5)
            )
            for token in query_tokens
        }
        idf_total = sum(query_idf.values()) or 1.0
        ranked: list[tuple[float, str, DocumentChunk, str]] = []
        for index in candidate_indices:
            chunk = self.chunks[index]
            corpus = _corpus_for(chunk)
            # Glossary is a display corpus, not a legal or contractual decision source.
            if corpus == "glossary":
                continue
            if product and product not in chunk.product and COMMON_PRODUCT not in chunk.product:
                continue
            if as_of and chunk.effective_from and chunk.effective_from > as_of:
                continue
            if as_of and chunk.effective_to and chunk.effective_to < as_of:
                continue

            tokens = self._token_sets[index]
            weighted_overlap = sum(query_idf[token] for token in query_tokens if token in tokens)
            text_score = weighted_overlap / idf_total if query_tokens else 0.0
            metadata_score = 0.0
            metadata_tokens = self._metadata_token_sets[index]
            metadata_overlap = sum(query_idf[token] for token in query_tokens if token in metadata_tokens)
            if metadata_overlap:
                metadata_score += min(0.22, (metadata_overlap / idf_total) * 0.18)
            compact_metadata = self._compact_metadata[index]
            if compact_metadata and compact_query:
                if compact_metadata in compact_query:
                    metadata_score += 0.24
                elif compact_query in compact_metadata:
                    metadata_score += 0.2
            matched_hints = [hint for hint in self._metadata_hints[index] if hint in compact_query]
            if matched_hints:
                longest_hint = max(len(hint) for hint in matched_hints)
                metadata_score += 0.85 if longest_hint >= 6 else 0.32
            if "상품설명서" in compact_query:
                if "상품설명서" in compact_metadata:
                    metadata_score += 0.18
                elif "특약" in compact_metadata:
                    metadata_score -= 0.12
            compact_overlap = sum(1 for term in compact_terms if term in self._compact_text[index])
            if compact_overlap:
                metadata_score += min(0.24, compact_overlap * 0.045)
            stem_score = 0.0
            if query_stems:
                stem_overlap = len(query_stems & self._stem_sets[index])
                if stem_overlap:
                    stem_score = min(0.3, (stem_overlap / len(query_stems)) * 0.4)
            vector_score = 0.0
            if query_embedding and chunk.embedding:
                vector_score = max(0.0, _cosine(query_embedding, chunk.embedding))
            if not text_score and not vector_score and not metadata_score and not stem_score:
                continue

            if text_score and vector_score:
                score = 0.7 * text_score + 0.3 * vector_score
                match_type = "hybrid"
            elif vector_score:
                score = vector_score
                match_type = "vector"
            else:
                score = text_score
                match_type = "full_text"
            if product and product in chunk.product:
                score += 0.18
            elif product and COMMON_PRODUCT in chunk.product:
                score += 0.04
            name_tokens = self._name_tokens[index]
            if name_tokens:
                name_overlap = query_tokens & name_tokens
                if name_overlap:
                    # 카테고리(예금/적금) 보너스보다 크게 줘서, 같은 카테고리
                    # 안에서도 정확히 이름이 겹치는 상품이 확실히 위로 온다.
                    score += 0.35 * (len(name_overlap) / len(name_tokens))
            score += metadata_score
            score += stem_score
            if intent == corpus:
                score += 0.45 if intent == "guides" else 0.14
            elif intent == "cases" and corpus == "products":
                score -= 0.04
            elif intent == "products" and corpus == "cases":
                score -= 0.08
            elif intent == "guides" and corpus in {"products", "cases", "regulations"}:
                score -= 0.35
            ranked.append((score, chunk.chunk_id, chunk, match_type))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        results: list[EvidenceRef] = []
        seen: set[str] = set()
        seen_chunks: set[str] = set()
        seen_docs: set[str] = set()

        def add_result(score: float, chunk: DocumentChunk, match_type: str, *, allow_same_doc: bool = False) -> bool:
            if chunk.chunk_id in seen_chunks:
                return False
            if not allow_same_doc and chunk.doc_id in seen_docs:
                return False
            key = _text_key(chunk.text)
            if key in seen:
                return False
            seen.add(key)
            seen_chunks.add(chunk.chunk_id)
            seen_docs.add(chunk.doc_id)
            results.append(
                EvidenceRef(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    path=chunk.path,
                    page=chunk.page,
                    section=chunk.section,
                    score=round(score, 4),
                    snippet=chunk.text[:280],
                    effective_from=chunk.effective_from,
                    effective_to=chunk.effective_to,
                    match_type=match_type,
                )
            )
            return len(results) >= top_k

        if intent:
            quota = max(1, math.ceil(top_k * 0.4))
            for score, _, chunk, match_type in ranked:
                if _corpus_for(chunk) != intent:
                    continue
                if add_result(score, chunk, match_type) or len(results) >= quota:
                    break

        for score, _, chunk, match_type in ranked:
            if add_result(score, chunk, match_type):
                break
            if len(results) >= top_k:
                break
        for score, _, chunk, match_type in ranked:
            if add_result(score, chunk, match_type, allow_same_doc=True):
                break
            if len(results) >= top_k:
                break
        return results

    def search_many(
        self,
        queries: Sequence[str],
        *,
        product: str | None = None,
        as_of: date | None = None,
        top_k: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[EvidenceRef]:
        """Fuse focused retrieval queries with reciprocal rank fusion."""
        unique_queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
        if not unique_queries:
            return []
        candidates: dict[str, tuple[EvidenceRef, float]] = {}
        per_query_limit = max(top_k * 4, 20)
        for query in unique_queries:
            for rank, evidence in enumerate(
                self.search(
                    query,
                    product=product,
                    as_of=as_of,
                    top_k=per_query_limit,
                    query_embedding=query_embedding,
                ),
                start=1,
            ):
                rrf = 1.0 / (RRF_K + rank)
                previous = candidates.get(evidence.chunk_id)
                if previous is None:
                    candidates[evidence.chunk_id] = (evidence, rrf)
                else:
                    best = previous[0] if previous[0].score >= evidence.score else evidence
                    candidates[evidence.chunk_id] = (best, previous[1] + rrf)

        if not candidates:
            return []
        max_rrf = max(value[1] for value in candidates.values())
        fused: list[tuple[float, EvidenceRef]] = []
        for evidence, rrf in candidates.values():
            score = 0.7 * evidence.score + 0.3 * (rrf / max_rrf)
            fused.append((score, evidence))
        fused.sort(key=lambda item: (-item[0], item[1].chunk_id))
        # 질의마다 같은 문구의 다른 사본이 1위로 뽑힐 수 있어 융합 후 한 번 더 거른다.
        results: list[EvidenceRef] = []
        seen: set[str] = set()
        for score, evidence in fused:
            key = _text_key(evidence.snippet)
            if key in seen:
                continue
            seen.add(key)
            results.append(evidence.model_copy(update={"score": round(score, 4)}))
            if len(results) >= top_k:
                break
        return results
