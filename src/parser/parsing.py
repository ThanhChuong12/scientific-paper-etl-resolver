import os
import re
import json
import hashlib
import shutil
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================ CONSTANTS AND CONFIGURATIONS ================
# LaTeX SECTION HIERARCHY
# Defines the logical depth of LaTeX sectioning commands.
SECTION_HIERARCHY = {
    'part': 0,
    'chapter': 1,
    'abstract': 1,
    'section': 2,    
    'subsection': 3,
    'subsubsection': 4,
    'paragraph': 5,
    'subparagraph': 6
}

# FORMATTING-ONLY COMMANDS
# These commands carry no semantic meaning and can be safely removed
FORMATTING_COMMANDS = [
    r'\\centering',
    r'\\raggedright',
    r'\\raggedleft',
    r'\\noindent',
    r'\\small',
    r'\\footnotesize',
    r'\\scriptsize',
    r'\\tiny',
    r'\\normalsize',
    r'\\large',
    r'\\Large',
    r'\\LARGE',
    r'\\huge',
    r'\\Huge',
    r'\\toprule',
    r'\\midrule',
    r'\\bottomrule',
    r'\\hline',
    r'\\cline\{[^}]*\}',
    r'\\newpage',
    r'\\clearpage',
    r'\\pagebreak',
    r'\\linebreak',
    r'\\hfill',
    r'\\vfill',
    r'\\hspace\*?\{[^}]*\}',
    r'\\vspace\*?\{[^}]*\}',
    r'\\bigskip',
    r'\\medskip',
    r'\\smallskip',
    r'\\par\b',
    r'\\indent',
    r'\\setlength\{[^}]*\}\{[^}]*\}',
    r'\\addtolength\{[^}]*\}\{[^}]*\}',
    r'\\label\{[^}]*\}',
]

# TEXT FORMATTING COMMANDS (CONTENT PRESERVING)
TEXT_FORMAT_COMMANDS = [
    (r'\\textbf\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1'),
    (r'\\textit\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1'),
    (r'\\emph\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1'),
    (r'\\underline\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1'),
    (r'\\texttt\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1'),
    (r'\\textsf\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1'),
    (r'\\textrm\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1'),
    (r'\\textsc\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1'),
    (r'\\textnormal\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1'),
    (r'\\mbox\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1'),
]

# NODE TYPE PREFIXES
TYPE_PREFIXES = {
    'root': 'doc_root',
    'part': 'part',
    'chapter': 'chapter',
    'section': 'section',
    'subsection': 'subsection',
    'subsubsection': 'subsubsection',
    'paragraph': 'paragraph',
    'subparagraph': 'subparagraph',
    'abstract': 'abstract',
    'itemize': 'itemize',
    'enumerate': 'enumerate',
    'item': 'item',
    'equation': 'eq',
    'figure': 'fig',
    'table': 'fig',
    'sentence': 'sent'
}

# NON-TERMINATING ABBREVIATIONS
# Abbreviations that should not be interpreted as sentence boundaries.
ABBREVIATIONS = [
    r'Fig\.', r'Figs\.', r'Eq\.', r'Eqs\.', r'Sec\.', r'Secs\.', 
    r'Ch\.', r'Tab\.', r'Ref\.', r'Refs\.', r'al\.', r'et al\.',
    r'i\.e\.', r'e\.g\.', r'vs\.', r'etc\.', r'cf\.', r'viz\.',
    r'Dr\.', r'Mr\.', r'Mrs\.', r'Ms\.', r'Prof\.', r'Jr\.', r'Sr\.',
    r'No\.', r'Vol\.', r'pp\.', r'ed\.', r'eds\.'
]

# Citation command pattern (handles \cite variants and \nocite)
CITE_COMMAND_PATTERN = re.compile(
    r'\\(?:[A-Za-z]*cite[a-zA-Z]*|nocite)\*?(?:\[[^\]]*\])*?\s*\{([^}]*)\}',
    re.DOTALL
)

# Files larger than this threshold (in bytes) are parsed with a streaming parser
LARGE_BIB_THRESHOLD_BYTES = 5 * 1024 * 1024  # 5 MB

# ================ DATA CLASSES ================
@dataclass
class HierarchyNode:
    """Represents a node in the document hierarchy."""
    id: str
    content: str
    node_type: str
    level: int
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    
    
@dataclass
class BibEntry:
    """Represents a bibliography entry."""
    key: str
    entry_type: str
    fields: Dict[str, str] = field(default_factory=dict)
    raw_content: str = ""
    content_hash: str = ""



# ================ MULTI-FILE GATHERING ================
class TexFileGatherer:
    """
    Responsible for gathering and merging LaTeX source files within a version directory.

    This class:
    - Scans for all `.tex` files
    - Heuristically identifies the main compilation file
    - Recursively resolves \\input and \\include directives
    - Tracks which files are actually used in compilation
    """

    def __init__(self, version_dir: Path) -> None:
        self.version_dir: Path = version_dir
        self.all_tex_files: List[Path] = []
        self.visited_files: Set[Path] = set()
        self.included_files: Set[Path] = set()
        
    # FILE DISCOVERY    
    def find_all_tex_files(self) -> List[Path]:
        self.all_tex_files = list(self.version_dir.rglob("*.tex"))
        return self.all_tex_files
    
    # MAIN FILE IDENTIFICATION
    def identify_main_file(self) -> Optional[Path]:
        """
        Identify the main LaTeX compilation file using heuristic scoring.

        Heuristics include:
        - Presence of \\documentclass and \\begin{document}
        - Common main file names (e.g., main.tex, paper.tex)
        - Penalization of likely sub-files (appendix, chapter, etc.)
        """
        if not self.all_tex_files:
            self.find_all_tex_files()

        candidates = []
        for tex_file in self.all_tex_files:
            try:
                content = tex_file.read_text(encoding="utf-8", errors="ignore")
                score = 0

                # Strong indicators of a main file
                if r"\documentclass" in content:
                    score += 10
                if r"\begin{document}" in content:
                    score += 10
                if r"\end{document}" in content:
                    score += 5

                # Secondary indicators
                if r"\maketitle" in content:
                    score += 3
                if r"\tableofcontents" in content:
                    score += 2
                if r"\bibliography" in content or r"\printbibliography" in content:
                    score += 2

                # Penalize files that look like partials
                filename_lower = tex_file.name.lower()
                if tex_file.name.startswith("_"):
                    score -= 5
                if any(token in filename_lower for token in ("appendix", "chapter", "section")):
                    score -= 2

                # Common main file naming patterns
                if tex_file.stem.lower() in {"main", "paper", "article", "thesis", "document"}:
                    score += 5
                if score > 0:
                    candidates.append((tex_file, score))
            except Exception as exc:
                logger.warning("Failed to read %s: %s", tex_file, exc)

        if not candidates:
            return self.all_tex_files[0] if self.all_tex_files else None
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0][0]
    
    # INCLUDE RESOLUTION    
    def resolve_includes(self, main_file: Path) -> str:
        """
        Resolve all \\input and \\include directives starting from the main file.
        """
        self.visited_files.clear()
        self.included_files.clear()
        return self._read_file_recursive(main_file)
    
    
    def _read_file_recursive(self, file_path: Path) -> str:
        """Recursively read a LaTeX file and resolve its includes."""
        if file_path in self.visited_files:
            return ""
            
        self.visited_files.add(file_path)
        self.included_files.add(file_path)
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"Cannot read {file_path}: {e}")
            return ""
        include_pattern = re.compile(
            r'\\(?:input|include)(?:\s*\{([^}]+)\}|\s+([^\s\\]+))',
            re.MULTILINE
        )

        def replace_include(match):
            """
            Replace an include directive with the resolved file content.
            """
            included_name = (match.group(1) or match.group(2)).strip()
            candidate_paths = [
                file_path.parent / included_name,
                file_path.parent / f"{included_name}.tex",
                self.version_dir / included_name,
                self.version_dir / f"{included_name}.tex",
            ]

            for candidate in candidate_paths:
                if candidate.exists() and candidate.is_file():
                    return self._read_file_recursive(candidate)
            logger.debug("Included file not found: %s", included_name)
            return ""

        return include_pattern.sub(replace_include, content)
    
    # UNUSED FILE DETECTION
    def get_unused_files(self) -> List[Path]:
        """Identify LaTeX files that were not included in the final compilation."""
        if not self.all_tex_files:
            self.find_all_tex_files()
        return [tex for tex in self.all_tex_files if tex not in self.included_files]

# ================ LATEX CLEANUP AND NORMALIZATION ================
class LatexCleaner:
    """
    Performs LaTeX cleanup and normalization prior to structural parsing.

    This class is responsible for:
    - Removing comments and layout-only commands
    - Simplifying text formatting while preserving content
    - Normalizing inline and block math expressions
    - Cleaning environment options (e.g., figure placement)
    - Normalizing whitespace for downstream tokenization
    """
        
    def __init__(self) -> None:
        self.abbreviation_placeholders: Dict[str, str] = {}

    # PUBLIC API
    def clean(self, content: str, strip_sectioning: bool = False) -> str:
        """Apply all cleanup and normalization pipeline.

        strip_sectioning: if True, remove section-like commands while preserving titles.
        Use this only for display/inspection; the parser should keep sectioning intact.
        """
        content = self.remove_comments(content)
        content = self.remove_formatting_commands(content)
        content = self.simplify_text_formatting(content)
        if strip_sectioning:
            content = self.remove_sectioning_commands(content)
        content = self.normalize_math(content)
        content = self.clean_environment_options(content)
        content = self.normalize_whitespace(content)
        return content

    # SECTIONING REMOVAL (TEXT-PRESERVING)
    def remove_sectioning_commands(self, content: str) -> str:
        """
        Strip section-like commands (including abstract) while preserving their titles.

        This prevents orphan braces (e.g., "Introduction}") in normalized text streams
        used for comparative inspection; avoid enabling during full parse to keep
        section markers intact for hierarchy building.
        """
        section_cmd = re.compile(
            r"\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph|abstract)\*?\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
            re.IGNORECASE
        )
        return section_cmd.sub(lambda m: m.group(2).strip() + "\n", content)
    
    # COMMENT REMOVAL
    def remove_comments(self, content: str) -> str:
        """Remove LaTeX comments while preserving document structure."""
        cleaned_lines = []
        for line in content.split("\n"):
            buffer = []
            index = 0
            while index < len(line):
                if line[index] == "%" and (index == 0 or line[index - 1] != "\\"):
                    break
                buffer.append(line[index])
                index += 1
            cleaned_line = "".join(buffer)
            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)
            else:
                cleaned_lines.append("")
        return "\n".join(cleaned_lines)

    # FORMATTING COMMAND REMOVAL
    def remove_formatting_commands(self, content: str) -> str:
        """Remove layout-only LaTeX commands with no semantic meaning."""
        for pattern in FORMATTING_COMMANDS:
            content = re.sub(pattern, "", content)
        return content
    
    # TEXT FORMATTING SIMPLIFICATION
    def simplify_text_formatting(self, content: str) -> str:
        """
        Simplify text formatting commands by stripping the command
        and preserving inner content.
        """
        for _ in range(3):
            for pattern, replacement in TEXT_FORMAT_COMMANDS:
                content = re.sub(pattern, replacement, content)
        return content
        
    # MATH NORMALIZATION
    def normalize_math(self, content: str) -> str:
        """
        Normalize all math expressions into a unified representation.

        Transformations:
        - Inline math: \\( ... \\) → $ ... $
        - Block math: \\[ ... \\], $$ ... $$ -> equation environment
        - Multi-line math environments (align, gather, etc.) -> equation
        """
        # Inline math: \( ... \) → $ ... $
        content = re.sub(
            r"\\\((.+?)\\\)",
            r"$\1$",
            content,
            flags=re.DOTALL,
        )

        def wrap_equation(match: re.Match) -> str:
            inner_content = match.group(1).strip()
            return (
                "\\begin{equation}\n"
                f"{inner_content}\n"
                "\\end{equation}"
            )

        # Block math: \[ ... \]
        content = re.sub(
            r"\\\[(.+?)\\\]",
            wrap_equation,
            content,
            flags=re.DOTALL,
        )

        # Block math: $$ ... $$ (excluding escaped dollars)
        content = re.sub(
            r"(?<!\\)\$\$(.+?)(?<!\\)\$\$",
            wrap_equation,
            content,
            flags=re.DOTALL,
        )

        # Other math environments normalized to equation
        math_environments = [
            "align", "align\\*", "gather", "gather\\*", "multline",
            "multline\\*", "alignat", "alignat\\*", "eqnarray", "eqnarray\\*",
            "displaymath", "flalign", "flalign\\*",
        ]

        for env in math_environments:
            escaped_env = env.replace("*", r"\*")
            pattern = re.compile(
                rf"\\begin\{{{escaped_env}\}}(.+?)\\end\{{{escaped_env}\}}",
                re.DOTALL,
            )
            content = pattern.sub(wrap_equation, content)

        return content
    
    # ENVIRONMENT CLEANUP
    def clean_environment_options(self, content: str) -> str:
        content = re.sub(
            r'(\\begin\{(?:figure|table|algorithm)\*?\})\s*\[[^\]]*\]',
            r'\1',
            content
        )
        return content
    
    # WHITESPACE NORMALIZATION
    def normalize_whitespace(self, content: str) -> str:
        """
        Normalize whitespace while preserving logical structure.
        
        Rules:
        - Collapse multiple spaces and tabs into one space
        - Limit consecutive blank lines to at most two
        - Trim leading and trailing whitespace
        """
        # Drop stray brace-only lines that can appear after command stripping
        content = re.sub(r'^\s*\}\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n\s*\n+", "\n\n", content)
        return content.strip()


# ================ REFERENCE EXTRACTION AND PROCESSING ================
class BibProcessor:
    """
    Handles bibliography extraction, normalization, and deduplication.

    Responsibilities:
    - Parse BibTeX (.bib) files
    - Extract \\bibitem entries from LaTeX sources
    - Normalize references into a unified BibEntry representation
    - Deduplicate references across document versions using content hashing
    - Normalize citation keys in LaTeX content
    """
    
    def __init__(self) -> None:
        self.entries: Dict[str, BibEntry] = {}
        self.content_hash_map: Dict[str, str] = {}   # content_hash -> canonical_key
        self.key_aliases: Dict[str, str] = {}        # old_key -> canonical_key

    # BIB FILE PARSING    
    def parse_bib_file(self, bib_path: Path, allowed_keys: Optional[Set[str]] = None) -> List[BibEntry]:
        """Parse a BibTeX (.bib) file into BibEntry objects."""
        entries: List[BibEntry] = []

        if allowed_keys is not None and not allowed_keys:
            logger.debug("Skipping %s because no cited keys target this file", bib_path)
            return entries

        selective_scan = (
            allowed_keys is not None
            and bib_path.exists()
            and bib_path.stat().st_size >= LARGE_BIB_THRESHOLD_BYTES
        )

        if selective_scan:
            return self._parse_bib_file_stream(bib_path, allowed_keys)

        try:
            content = bib_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("Cannot read bib file %s: %s", bib_path, exc)
            return entries

        entry_pattern = re.compile(
            r"@(\w+)\s*\{\s*([^,]+)\s*,(.*?)\}\s*(?=@|$)",
            re.DOTALL,
        )

        remaining = set(allowed_keys) if allowed_keys is not None else None

        for match in entry_pattern.finditer(content):
            entry_type = match.group(1).lower()
            key = match.group(2).strip()
            fields_block = match.group(3)

            if remaining is not None and key not in remaining:
                continue

            entry = self._build_entry(entry_type, key, fields_block, match.group(0))
            if entry:
                entries.append(entry)

            if remaining is not None:
                remaining.discard(key)
                if not remaining:
                    break

        return entries

    def _parse_bib_file_stream(self, bib_path: Path, allowed_keys: Set[str]) -> List[BibEntry]:
        """Stream a large BibTeX file and keep only the cited keys."""
        entries: List[BibEntry] = []
        target_keys = set(allowed_keys or [])
        if not target_keys:
            return entries

        entry_start_pattern = re.compile(r'@\s*([A-Za-z]+)\s*\{\s*([^,]+)\s*,')

        try:
            file_obj = bib_path.open("r", encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("Cannot read bib file %s: %s", bib_path, exc)
            return entries

        with file_obj:
            inside_entry = False
            capture_entry = False
            brace_depth = 0
            buffer: List[str] = []
            current_type = ""
            current_key = ""
            resolved_keys: Set[str] = set()

            for line in file_obj:
                stripped = line.lstrip()

                if not inside_entry:
                    match = entry_start_pattern.match(stripped)
                    if not match:
                        continue

                    inside_entry = True
                    current_type = match.group(1).lower()
                    current_key = match.group(2).strip()
                    capture_entry = current_key in target_keys
                    brace_depth = stripped.count('{') - stripped.count('}')

                    if capture_entry:
                        buffer = [line]
                    else:
                        buffer = []

                    if brace_depth <= 0:
                        if capture_entry:
                            entry = self._finalize_stream_entry(buffer, current_type, current_key)
                            if entry:
                                entries.append(entry)
                                resolved_keys.add(entry.key)
                                if resolved_keys >= target_keys:
                                    break
                        inside_entry = False
                        capture_entry = False
                        buffer = []
                    continue

                brace_depth += line.count('{') - line.count('}')
                if capture_entry:
                    buffer.append(line)

                if brace_depth <= 0:
                    inside_entry = False
                    if capture_entry:
                        entry = self._finalize_stream_entry(buffer, current_type, current_key)
                        if entry:
                            entries.append(entry)
                            resolved_keys.add(entry.key)
                            if resolved_keys >= target_keys:
                                break
                    capture_entry = False
                    buffer = []

        return entries

    def _finalize_stream_entry(
        self,
        buffer: List[str],
        entry_type: str,
        key: str
    ) -> Optional[BibEntry]:
        """Construct a BibEntry from streamed lines."""
        raw_entry = ''.join(buffer)
        fields_block = self._extract_fields_block(raw_entry)
        return self._build_entry(entry_type, key, fields_block, raw_entry)

    def _extract_fields_block(self, raw_entry: str) -> str:
        """Return the field block substring from a full BibTeX entry."""
        at_index = raw_entry.find('{')
        if at_index == -1:
            return ""
        comma_index = raw_entry.find(',', at_index)
        if comma_index == -1:
            return ""
        fields_segment = raw_entry[comma_index + 1:]
        closing_index = fields_segment.rfind('}')
        if closing_index != -1:
            fields_segment = fields_segment[:closing_index]
        return fields_segment

    def _build_entry(
        self,
        entry_type: str,
        key: str,
        fields_block: str,
        raw_content: str
    ) -> Optional[BibEntry]:
        if not fields_block:
            return None

        fields = self._parse_bib_fields(fields_block)
        content_hash = self._compute_content_hash(fields)
        return BibEntry(
            key=key,
            entry_type=entry_type,
            fields=fields,
            raw_content=raw_content,
            content_hash=content_hash,
        )

    @staticmethod
    def extract_citation_keys(content: str) -> Tuple[Set[str], bool]:
        """Return cited keys and whether \nocite{*} is present."""
        keys: Set[str] = set()
        include_all = False

        for match in CITE_COMMAND_PATTERN.finditer(content):
            block = match.group(1)
            for raw_key in block.split(','):
                cleaned = raw_key.strip()
                if not cleaned:
                    continue
                if cleaned == '*':
                    include_all = True
                else:
                    keys.add(cleaned)

        return keys, include_all
    
    # BIBITEM EXTRACTION FROM LATEX
    def extract_bibitems(self, content: str) -> List[BibEntry]:
        """
        Extract \\bibitem entries from LaTeX content and convert them
        into BibEntry objects.
        """
        entries: List[BibEntry] = []

        bibitem_pattern = re.compile(
            r"\\bibitem(?:\[([^\]]*)\])?\{([^}]+)\}(.+?)"
            r"(?=\\bibitem|\\end\{thebibliography\}|$)",
            re.DOTALL,
        )

        for match in bibitem_pattern.finditer(content):
            key = match.group(2).strip()
            raw_content = match.group(3).strip()

            fields = self._parse_bibitem_content(raw_content)
            content_hash = self._compute_content_hash(fields)

            entries.append(
                BibEntry(
                    key=key,
                    entry_type=fields.get("_type", "misc"),
                    fields={k: v for k, v in fields.items() if not k.startswith("_")},
                    raw_content=raw_content,
                    content_hash=content_hash,
                )
            )

        return entries
    
    # INTERNAL UTILITIES 
    def _parse_bib_fields(self, fields_block: str) -> Dict[str, str]:
        """Parse BibTeX field definitions into a dictionary."""
        fields: Dict[str, str] = {}

        field_pattern = re.compile(
            r"(\w+)\s*=\s*(?:"
            r"\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
            r"|\"([^\"]*)\""
            r"|(\d+))",
            re.DOTALL,
        )

        for match in field_pattern.finditer(fields_block):
            field_name = match.group(1).lower()
            value = match.group(2) or match.group(3) or match.group(4) or ""
            fields[field_name] = self._clean_bib_value(value)
        return fields

    def _clean_bib_value(self, value: str) -> str:
        """Normalize BibTeX field values by stripping wrappers and collapsing whitespace."""
        cleaned = value.strip()
        # Remove leading/trailing braces or quotes (including double-braced titles)
        while len(cleaned) > 1 and (
            (cleaned.startswith("{") and cleaned.endswith("}")) or
            (cleaned.startswith("\"") and cleaned.endswith("\""))
        ):
            cleaned = cleaned[1:-1].strip()
        cleaned = " ".join(cleaned.split())
        return cleaned
    
    def _parse_bibitem_content(self, content: str) -> Dict[str, str]:
        """Heuristically extract structured fields from a \\bibitem entry."""

        fields: Dict[str, str] = {"_type": "misc"}
        content = content.strip()

        # Preserve raw content for traceability
        fields["note"] = content

        author_match = re.match(r"^([^.]+?)\.", content)
        if author_match:
            fields["author"] = author_match.group(1).strip()

        year_match = re.search(r"\b(19|20)\d{2}\b", content)
        if year_match:
            fields["year"] = year_match.group(0)

        title_match = re.search(r"[\"']([^\"']+)[\"']", content)
        if title_match:
            fields["title"] = title_match.group(1)

        return fields
    
    # DEDUPLICATION UTILITIES
    def _compute_content_hash(self, fields: Dict[str, str]) -> str:
        """
        Compute a stable hash used for cross-version deduplication.
        Normalization strategy:
        - Ignore internal fields (prefixed with '_')
        - Lowercase all values
        - Collapse extra whitespace
        - Sort keys for deterministic hashing
        """
        normalized: Dict[str, str] = {}

        for key, value in fields.items():
            if key.startswith("_"):
                continue
            normalized[key.lower()] = " ".join(value.lower().split())

        serialized = json.dumps(normalized, sort_keys=True)
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()
    
    # ENTRY REGISTRATION AND MERGING
    def add_entries(self, entries: List[BibEntry]) -> Dict[str, str]:
        """
        Register bibliography entries with content-based deduplication.

        Strategy:
        - Entries with identical content hashes are merged
        - Later versions overwrite earlier field values
        - BibTeX key collisions are resolved deterministically
        """
        key_mapping: Dict[str, str] = {}

        for entry in entries:
            if entry.content_hash in self.content_hash_map:
                canonical_key = self.content_hash_map[entry.content_hash]
                canonical_entry = self.entries[canonical_key]

                for field, value in entry.fields.items():
                    if not value:
                        continue
                    if field not in canonical_entry.fields:
                        canonical_entry.fields[field] = value
                    elif canonical_entry.fields[field] != value:
                        canonical_entry.fields[field] = value

                key_mapping[entry.key] = canonical_key
                self.key_aliases[entry.key] = canonical_key

            else:
                if entry.key in self.entries:
                    entry.key = f"{entry.key}_{entry.content_hash[:6]}"

                self.entries[entry.key] = entry
                self.content_hash_map[entry.content_hash] = entry.key
                key_mapping[entry.key] = entry.key

        return key_mapping
    
    # CITATION NORMALIZATION
    def normalize_citations(self, content: str) -> str:
        """Replace citation keys in LaTeX content with canonical keys."""
        def replace_citation(match: re.Match) -> str:
            command = match.group(1)
            keys = [k.strip() for k in match.group(2).split(",")]

            canonical_keys = []
            seen = set()

            for key in keys:
                canonical = self.key_aliases.get(key, key)
                if canonical not in seen:
                    canonical_keys.append(canonical)
                    seen.add(canonical)

            return f"\\{command}{{{','.join(canonical_keys)}}}"

        return re.sub(
            r"\\(cite[a-z]*)\{([^}]+)\}",
            replace_citation,
            content,
        )
    
    # EXPORT
    def export_bib(self, output_path: Path) -> None:
        """
        Export all canonical bibliography entries to a BibTeX file.
        """
        with output_path.open("w", encoding="utf-8") as file:
            for key, entry in sorted(self.entries.items()):
                file.write(f"@{entry.entry_type}{{{key},\n")
                for field, value in sorted(entry.fields.items()):
                    if value:
                        escaped_value = value.replace("\\", "\\\\")
                        file.write(f"  {field} = {{{escaped_value}}},\n")
                file.write("}\n\n")


# ================ HIERARCHY CONSTRUCTION ================
class HierarchyBuilder:
    """
    Build a hierarchical tree structure from LaTeX document content.

    The builder parses sections, lists, figures, equations, and sentences,
    then constructs a parent-child hierarchy suitable for downstream indexing
    or semantic processing.
    """
    def __init__(self, paper_id: str, version: str):
        self.paper_id = paper_id
        self.version = version
        self.elements: Dict[str, str] = {} # element_id -> raw content
        self.hierarchy: Dict[str, str] = {} # child_id -> parent_id
        self.content_hash_map: Dict[str, str] = {} # content_hash -> element_id 

    # PUBLIC API
    def build(self, content: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Build hierarchy from LaTeX content.
        """
        document_body = self._extract_document_body(content)
        document_body = self._remove_references_section(document_body)

        root_id = f"{self.paper_id}_doc_root"
        self.elements[root_id] = "DOCUMENT_ROOT"

        self._parse_structure(document_body, root_id)
        return self.elements, self.hierarchy
    
    # DOCUMENT PREPROCESS
    def _extract_document_body(self, content: str) -> str:
        """Extract content between \\begin{document} and \\end{document}."""
        match = re.search(
            r'\\begin\{document\}(.+?)\\end\{document\}',
            content,
            re.DOTALL
        )
        if match:
            return match.group(1)
        return content
    
    def _remove_references_section(self, content: str) -> str:
        """Remove References / Bibliography sections and environments."""
        lines = content.splitlines()
        output_lines = []
        in_references = False

        reference_header_pattern = re.compile(
            r'\\(section|chapter)\*?\{[^}]*(references|bibliography|references and notes)[^}]*\}',
            re.IGNORECASE
        )
        section_start_pattern = re.compile(r'\\(section|chapter)\*?\{')

        for line in lines:
            if reference_header_pattern.search(line):
                in_references = True
                continue
            if in_references:
                if section_start_pattern.match(line) and not reference_header_pattern.search(line):
                    in_references = False
                    output_lines.append(line)
            else:
                output_lines.append(line)
        cleaned_text = "\n".join(output_lines)

        # Remove thebibliography environment entirely
        cleaned_text = re.sub(
            r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
            '',
            cleaned_text,
            flags=re.DOTALL
        )
        return cleaned_text
    
    # STRUCTURE PARSING
    def _parse_structure(self, content: str, root_id: str) -> None:
        """
        Parse LaTeX content and construct the hierarchy.
        """
        stack = [{'id': root_id, 'level': -1, 'type': 'root'}]
        text_buffer: List[str] = []
        section_pattern = re.compile(
            r'\\(part|chapter|abstract|section|subsection|subsubsection|paragraph|subparagraph)\*?'
            r'\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        )

        lines = content.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # ---------- FIGURES / TABLES / ALGORITHMS ----------
            fig_match = re.match(r'\\begin\{(figure|table|algorithm)\*?\}', line)
            if fig_match:
                self._flush_text_buffer(text_buffer, stack[-1]['id'])
                text_buffer.clear()
                env_type = fig_match.group(1)
                block, end_i = self._extract_environment(lines, i, env_type)
                elem_id = self._create_element_id('fig', block)
                self._register_element(elem_id, block, stack[-1]['id'])
                i = end_i + 1
                continue

            # ------------------- EQUATIONS ---------------------
            eq_match = re.match(r'\\begin\{(equation|align|gather|multline)\*?\}', line)
            if eq_match:
                self._flush_text_buffer(text_buffer, stack[-1]['id'])
                text_buffer.clear()
                env_type = eq_match.group(1)
                block, end_i = self._extract_environment(lines, i, env_type)
                elem_id = self._create_element_id('eq', block)
                self._register_element(elem_id, block, stack[-1]['id'])
                i = end_i + 1
                continue

            # -------------------- ABSTRACT ---------------------
            if re.match(r'\\begin\{abstract\}', line):
                self._flush_text_buffer(text_buffer, stack[-1]['id'])
                text_buffer.clear()
                block, end_i = self._extract_environment(lines, i, 'abstract')
                abstract_text = self._strip_environment_content(block, 'abstract')
                abstract_id = self._create_element_id('abstract', 'abstract')
                self._register_element(abstract_id, 'abstract', stack[-1]['id'])
                self._flush_text_buffer([abstract_text], abstract_id)
                i = end_i + 1
                continue

            # -------------------- SECTIONS ---------------------
            sec_match = section_pattern.match(line)
            if sec_match:
                self._flush_text_buffer(text_buffer, stack[-1]['id'])
                text_buffer.clear()
                sec_type, title = sec_match.groups()
                level = SECTION_HIERARCHY.get(sec_type, 6)
                while len(stack) > 1 and (
                    stack[-1]['type'] in {'item', 'itemize', 'enumerate'} or
                    stack[-1]['level'] >= level
                ):
                    stack.pop()
                sec_id = self._create_element_id(sec_type, title)
                self._register_element(sec_id, title, stack[-1]['id'])
                stack.append({'id': sec_id, 'level': level, 'type': sec_type})
                i += 1
                continue

            # --------------------- LISTS -----------------------
            if '\\begin{itemize}' in line or '\\begin{enumerate}' in line:
                self._flush_text_buffer(text_buffer, stack[-1]['id'])
                text_buffer.clear()
                list_type = 'itemize' if 'itemize' in line else 'enumerate'
                list_id = self._create_element_id(list_type, f"list_{i}")
                self._register_element(list_id, list_type, stack[-1]['id'])
                stack.append({
                    'id': list_id,
                    'level': stack[-1]['level'] + 0.5,
                    'type': list_type
                })
                i += 1
                continue

            if '\\end{itemize}' in line or '\\end{enumerate}' in line:
                self._flush_text_buffer(text_buffer, stack[-1]['id'])
                text_buffer.clear()
                if stack[-1]['type'] == 'item':
                    stack.pop()
                if stack[-1]['type'] in {'itemize', 'enumerate'}:
                    stack.pop()
                i += 1
                continue

            # ---------------------- ITEMS ----------------------
            if line.startswith('\\item'):
                self._flush_text_buffer(text_buffer, stack[-1]['id'])
                text_buffer.clear()
                if stack[-1]['type'] == 'item':
                    stack.pop()
                if stack[-1]['type'] in {'itemize', 'enumerate'}:
                    item_text = line[5:].strip()
                    item_id = self._create_element_id('item', f"item_{i}")
                    self._register_element(item_id, "item", stack[-1]['id'])
                    stack.append({
                        'id': item_id,
                        'level': stack[-1]['level'] + 0.5,
                        'type': 'item'
                    })
                    if item_text:
                        text_buffer.append(item_text)
                else:
                    text_buffer.append(line)
                i += 1
                continue

            # ------------------ REGULAR TEXT -------------------
            text_buffer.append(line)
            i += 1
        self._flush_text_buffer(text_buffer, stack[-1]['id'])

    # ========================== UTILITIES ===============================
    def _extract_environment(self, lines: List[str], start_index: int, env_type: str) -> Tuple[str, int]:
        """Extract a full LaTeX environment block."""
        collected = []
        depth = 0
        i = start_index
        env_pattern = env_type.replace('*', r'\*?')

        while i < len(lines):
            line = lines[i]
            if re.search(fr'\\begin\{{{env_pattern}\}}', line):
                depth += 1
            if re.search(fr'\\end\{{{env_pattern}\}}', line):
                depth -= 1
            collected.append(line)
            if depth == 0:
                break
            i += 1
        return "\n".join(collected), i

    def _strip_environment_content(self, block: str, env_type: str) -> str:
        """Remove \begin/\end wrappers from an environment block."""
        pattern = re.compile(
            rf"\\begin\{{{re.escape(env_type)}\}}(.*?)\\end\{{{re.escape(env_type)}\}}",
            re.DOTALL
        )
        match = pattern.search(block)
        return match.group(1).strip() if match else block
    
    def _flush_text_buffer(self, buffer: List[str], parent_id: str) -> None:
        """
        Convert accumulated text into sentence-level nodes.
        """
        if not buffer:
            return

        text = " ".join(buffer)
        text = " ".join(text.split())
        if not text:
            return
        for sentence in self._split_sentences(text):
            sent_id = self._create_element_id('sent', sentence)
            self._register_element(sent_id, sentence, parent_id)

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while protecting abbreviations.
        """
        protected_text = text
        placeholder_map = {}

        for i, abbr in enumerate(ABBREVIATIONS):
            placeholder = f"__ABBR{i}__"
            protected_text = re.sub(
                abbr,
                placeholder.replace('.', '<DOT>'),
                protected_text
            )
            placeholder_map[placeholder.replace('.', '<DOT>')] = abbr.replace('\\', '')
        chunks = re.split(r'(?<=[.!?])\s+', protected_text)

        sentences = []
        for chunk in chunks:
            restored = chunk
            for placeholder, original in placeholder_map.items():
                restored = restored.replace(placeholder, original.replace('<DOT>', '.'))
            restored = restored.replace('<DOT>', '.')
            if restored.strip():
                sentences.append(restored.strip())
        return sentences

    def _create_element_id(self, element_type: str, content: str) -> str:
        """
        Create a deterministic element ID using content hash.
        """
        prefix = TYPE_PREFIXES.get(element_type, element_type)
        digest = hashlib.md5(content.strip().encode("utf-8")).hexdigest()[:8]
        return f"{self.paper_id}_{prefix}_{digest}"

    def _register_element(self, element_id: str, content: str, parent_id: str) -> None:
        """
        Register an element and resolve ID collisions safely.
        """
        final_id = element_id
        counter = 1
        while final_id in self.hierarchy and self.hierarchy[final_id] != parent_id:
            final_id = f"{element_id}_{counter}"
            counter += 1
        self.elements[final_id] = content
        self.hierarchy[final_id] = parent_id

# ================ CONTENT DEDUPLICATION ACROSS VERSIONS ================
class ContentDeduplicator:
    """
    Deduplicate content across multiple document versions.

    Identical content blocks (after normalization) are merged into a single
    canonical element ID, while preserving version-specific hierarchies.
    """

    def __init__(self, paper_id: str):
        self.paper_id = paper_id
        self.global_elements: Dict[str, str] = {} # canonical_element_id -> original content
        self.version_hierarchies: Dict[str, Dict[str, str]] = {} # version -> {child_id -> parent_id}
        self.content_hash_to_id: Dict[str, str] = {} # content_hash -> canonical_element_id
        self.global_parent_links: Dict[str, Set[str]] = defaultdict(set) # canonical_child_id -> set(canonical_parent_id)
        
    # PUBLIC API
    def add_version(self, version: str, elements: Dict[str, str], hierarchy: Dict[str, str]) -> None:
        """Add a document version and merge its content with global storage."""
        id_mapping: Dict[str, str] = {}

        # Deduplicate Elements
        for original_id, content in elements.items():
            normalized_content = self._normalize_content(content)
            content_hash = self._hash_content(normalized_content)
            if content_hash in self.content_hash_to_id:
                canonical_id = self.content_hash_to_id[content_hash]
            else:
                canonical_id = original_id
                self.content_hash_to_id[content_hash] = canonical_id
                self.global_elements[canonical_id] = content
            id_mapping[original_id] = canonical_id

        # Remap Hierarchy 
        remapped_hierarchy: Dict[str, str] = {}
        for child_id, parent_id in hierarchy.items():
            canonical_child = id_mapping.get(child_id, child_id)
            canonical_parent = id_mapping.get(parent_id, parent_id)
            remapped_hierarchy[canonical_child] = canonical_parent
            if canonical_parent:
                self.global_parent_links[canonical_child].add(canonical_parent)
        self.version_hierarchies[version] = remapped_hierarchy
    
    def get_merged_output(self) -> Dict[str, Any]:
        return {
            "elements": self.global_elements,
            "hierarchy": self.version_hierarchies,
            "merged_parents": {
                child_id: sorted(parent_ids)
                for child_id, parent_ids in self.global_parent_links.items()
            }
        }
    
    # INTERNAL UTILITIES
    def _normalize_content(self, content: str) -> str:
        """Normalize content for semantic comparison."""
        text = " ".join(content.split())
        text = text.lower()
        # Remove LaTeX commands like \textbf{...}, \emph{...}
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        # Remove remaining LaTeX syntax characters
        text = re.sub(r'[{}\\]', '', text)
        return text.strip()

    def _hash_content(self, normalized_content: str) -> str:
        return hashlib.md5(normalized_content.encode("utf-8")).hexdigest()


# ================ PAPER PROCESSOR ================
class PaperProcessor:
    """
    Main processor for a single paper.
    Orchestrates all versions and produces the final merged output.
    """
    
    def __init__(
        self,
        paper_dir: Path,
        output_dir: Path,
        skip_existing: bool = False
    ):
        self.paper_dir = paper_dir
        self.paper_id = paper_dir.name
        self.output_dir = output_dir / self.paper_id
        self.skip_existing = skip_existing
        self.cleaner = LatexCleaner()
        self.bib_processor = BibProcessor()
        self.deduplicator = ContentDeduplicator(self.paper_id)
        self.last_status: str = "pending"
        
    def process(self) -> bool:
        """Process all available versions of the paper."""
        self.last_status = "pending"
        tex_root = self.paper_dir / "tex"

        if not tex_root.exists():
            logger.warning(f"No 'tex' directory found for paper {self.paper_id}")
            self.last_status = "skipped"
            return False
        logger.info(f"Processing paper: {self.paper_id}")

        if self.skip_existing and self._outputs_exist():
            logger.info(
                f"  Skipping paper {self.paper_id} (existing outputs, --skip-existing)"
            )
            self.last_status = "skipped"
            return True

        version_dirs = self._discover_versions(tex_root)
        if not version_dirs:
            logger.warning(f"No version directories found in {tex_root}")
            self.last_status = "skipped"
            return False
        for version_dir in version_dirs:
            version_key = self._extract_version_key(version_dir.name)
            logger.info(f"  Processing version: {version_dir.name}")
            try:
                self._process_version(version_dir, version_key)
            except Exception as exc:
                logger.error(f"  Failed to process version {version_dir.name}: {exc}")
                import traceback
                traceback.print_exc()
        self._save_outputs()
        self._log_statistics()
        self.last_status = "success"
        return True
    
    def _discover_versions(self, tex_root: Path) -> List[Path]:
        """
        Discover version directories under the tex root.
        """
        versions = [
            d for d in tex_root.iterdir()
            if d.is_dir() and re.match(rf'{re.escape(self.paper_id)}v\d+', d.name)
        ]
        if not versions:
            versions = [
                d for d in tex_root.iterdir()
                if d.is_dir() and 'v' in d.name.lower()
            ]
        return sorted(versions)
    
    @staticmethod
    def _extract_version_key(version_name: str) -> str:
        """
        Extract numeric version key from directory name.
        """
        match = re.search(r'v(\d+)', version_name.lower())
        return match.group(1) if match else version_name

    def _process_version(self, version_dir: Path, version_key: str):
        """
        Process a single paper version.
        """
        gatherer = TexFileGatherer(version_dir)
        main_file = gatherer.identify_main_file()
        if not main_file:
            logger.warning(f"    No main TeX file found in {version_dir}")
            return
        logger.info(f"    Main file: {main_file.name}")
        merged_content = gatherer.resolve_includes(main_file)
        unused_files = gatherer.get_unused_files()
        if unused_files:
            logger.debug(
                f"    Unused TeX files: {[f.name for f in unused_files]}"
            )
        citation_keys, include_all_refs = self.bib_processor.extract_citation_keys(merged_content)
        self._process_bibliography(version_dir, merged_content, citation_keys, include_all_refs)
        cleaned_content = self.cleaner.clean(merged_content)
        cleaned_content = self.bib_processor.normalize_citations(cleaned_content)
        builder = HierarchyBuilder(self.paper_id, version_key)
        elements, hierarchy = builder.build(cleaned_content)
        self.deduplicator.add_version(version_key, elements, hierarchy)

    def _process_bibliography(
        self,
        version_dir: Path,
        content: str,
        citation_keys: Set[str],
        include_all_refs: bool
    ) -> None:
        """
        Process bibliography sources for a version.
        """
        remaining_keys = None if include_all_refs else set(citation_keys)

        for bib_file in version_dir.rglob("*.bib"):
            allowed_keys = None if remaining_keys is None else remaining_keys
            entries = self.bib_processor.parse_bib_file(bib_file, allowed_keys)
            if not entries:
                continue
            self.bib_processor.add_entries(entries)
            if remaining_keys is not None:
                resolved_keys = {entry.key for entry in entries}
                remaining_keys.difference_update(resolved_keys)
                if not remaining_keys:
                    break
        bibitem_entries = self.bib_processor.extract_bibitems(content)
        self.bib_processor.add_entries(bibitem_entries)

    def _outputs_exist(self) -> bool:
        """Return True if this paper already has generated outputs."""
        hierarchy_path = self.output_dir / "hierarchy.json"
        return hierarchy_path.exists()

    def _save_outputs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        merged_output = self.deduplicator.get_merged_output()
        hierarchy_path = self.output_dir / "hierarchy.json"
        with hierarchy_path.open("w", encoding="utf-8") as f:
            json.dump(merged_output, f, indent=2, ensure_ascii=False)
        if self.bib_processor.entries:
            self.bib_processor.export_bib(self.output_dir / "refs.bib")
        self._copy_optional_file("metadata.json")
        self._copy_optional_file("references.json")
        logger.info(f"  Outputs saved to {self.output_dir}")

    def _copy_optional_file(self, filename: str):
        """
        Copy optional metadata/reference files if present.
        """
        src = self.paper_dir / filename
        if src.exists():
            shutil.copy2(src, self.output_dir / filename)

    def _log_statistics(self):
        unique_elements = len(self.deduplicator.global_elements)
        raw_elements_estimate = unique_elements + sum(
            len(h) for h in self.deduplicator.version_hierarchies.values()
        )
        logger.info(f"  Statistics for paper {self.paper_id}:")
        logger.info(f"    - Raw elements (estimated): {raw_elements_estimate}")
        logger.info(f"    - Unique elements: {unique_elements}")

# ================ BATCH PROCESSOR ================
class BatchProcessor:
    """
    Batch processor for multiple papers.
    """

    def __init__(
        self,
        base_dir: Path,
        output_dir: Optional[Path] = None,
        max_workers: Optional[int] = None,
        skip_existing: bool = False
    ):
        self.base_dir = base_dir
        self.output_dir = output_dir or (base_dir / "processed")
        self.max_workers = max_workers
        self.skip_existing = skip_existing

    def run(self) -> Dict[str, List[str]]:
        paper_dirs = self._discover_papers()
        logger.info(f"Found {len(paper_dirs)} papers to process")

        stats = {
            "success": [],
            "failed": [],
            "skipped": []
        }
        if not paper_dirs:
            self._save_summary(stats)
            self._log_summary(stats)
            return stats

        worker_count = self._resolve_worker_count(len(paper_dirs))
        if worker_count > 1 and len(paper_dirs) > 1:
            logger.info(
                f"Running batch in parallel with {worker_count} worker(s)"
            )
            self._run_parallel(paper_dirs, worker_count, stats)
        else:
            self._run_sequential(paper_dirs, stats)
        self._save_summary(stats)
        self._log_summary(stats)
        return stats

    def _discover_papers(self) -> List[Path]:
        return sorted(
            d for d in self.base_dir.iterdir()
            if d.is_dir() and re.match(r'\d{4}[-.]?\d+', d.name)
        )

    def _resolve_worker_count(self, total_papers: int) -> int:
        if total_papers <= 1:
            return 1
        if self.max_workers is not None:
            return max(1, min(self.max_workers, total_papers))
        cpu_total = os.cpu_count() or 1
        suggested = max(1, cpu_total - 1)
        return max(1, min(suggested, total_papers))

    def _run_sequential(
        self,
        paper_dirs: List[Path],
        stats: Dict[str, List[str]]
    ) -> None:
        for paper_dir in paper_dirs:
            try:
                processor = PaperProcessor(
                    paper_dir,
                    self.output_dir,
                    skip_existing=self.skip_existing
                )
                result = processor.process()
                status = processor.last_status
                if status not in stats:
                    status = "success" if result else "skipped"
                stats[status].append(paper_dir.name)
            except Exception as exc:
                logger.error(f"Error processing paper {paper_dir.name}: {exc}")
                stats["failed"].append(paper_dir.name)
                import traceback
                traceback.print_exc()

    def _run_parallel(
        self,
        paper_dirs: List[Path],
        worker_count: int,
        stats: Dict[str, List[str]]
    ) -> None:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_name = {}
            for paper_dir in paper_dirs:
                future = executor.submit(
                    _process_paper_task,
                    str(paper_dir),
                    str(self.output_dir),
                    self.skip_existing
                )
                future_to_name[future] = paper_dir.name

            for future in as_completed(future_to_name):
                paper_name = future_to_name[future]
                try:
                    status, result_name, error = future.result()
                except Exception as exc:
                    logger.error(
                        f"Worker crashed while processing {paper_name}: {exc}"
                    )
                    stats["failed"].append(paper_name)
                    continue

                target_name = result_name or paper_name
                if status not in stats:
                    status = "failed" if error else "success"
                stats[status].append(target_name)
                if error:
                    logger.error(
                        f"Error processing paper {target_name}: {error}"
                    )

    def _save_summary(self, stats: Dict[str, List[str]]):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.output_dir / "processing_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    @staticmethod
    def _log_summary(stats: Dict[str, List[str]]):
        logger.info("Batch processing complete:")
        logger.info(f"  Success: {len(stats['success'])}")
        logger.info(f"  Skipped: {len(stats['skipped'])}")
        logger.info(f"  Failed: {len(stats['failed'])}")


def _process_paper_task(
    paper_dir_str: str,
    output_root_str: str,
    skip_existing: bool
) -> Tuple[str, str, Optional[str]]:
    """Worker entry point for parallel paper processing."""
    paper_dir = Path(paper_dir_str)
    output_root = Path(output_root_str)
    processor = PaperProcessor(
        paper_dir,
        output_root,
        skip_existing=skip_existing
    )
    try:
        result = processor.process()
        status = processor.last_status
        if status not in {"success", "skipped"}:
            status = "success" if result else "skipped"
        return status, paper_dir.name, None
    except Exception as exc:
        logger.error(f"Unhandled error in worker for {paper_dir.name}: {exc}")
        traceback.print_exc()
        return "failed", paper_dir.name, str(exc)


# ================ MAIN ENTRY POINT ================
def main() -> int:
    """
    Main entry point for the LaTeX parser.
    Handles CLI argument parsing and dispatches processing tasks.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="LaTeX Parser for Scientific Publications"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/ColabProject_5000/23120195"),
        help="Base directory containing paper directories"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory (default: <input_dir>/processed)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )

    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Process a single paper by paper ID"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip papers that already have generated outputs"
    )

    args, _ = parser.parse_known_args()
    _configure_logging(args.verbose)
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    output_dir = args.output or (args.input_dir / "processed")
    if args.single:
        return _run_single_paper(
            args.input_dir,
            output_dir,
            args.single,
            skip_existing=args.skip_existing
        )
    return _run_batch(
        args.input_dir,
        output_dir,
        max_workers=args.workers,
        skip_existing=args.skip_existing
    )


def _configure_logging(verbose: bool):
    """
    Configure global logging level.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

def _run_single_paper(
    base_dir: Path,
    output_dir: Path,
    paper_id: str,
    skip_existing: bool = False
) -> int:
    """
    Process a single paper.
    """
    paper_dir = base_dir / paper_id

    if not paper_dir.exists():
        logger.error(f"Paper directory does not exist: {paper_dir}")
        return 1

    logger.info(f"Processing single paper: {paper_id}")
    processor = PaperProcessor(
        paper_dir,
        output_dir,
        skip_existing=skip_existing
    )
    if processor.process():
        if processor.last_status == "skipped":
            logger.info(
                "Paper already processed; skipping per --skip-existing"
            )
        else:
            logger.info("Processing completed successfully")
        return 0
    logger.error("Processing failed")
    return 1

def _run_batch(
    base_dir: Path,
    output_dir: Path,
    max_workers: Optional[int],
    skip_existing: bool
) -> int:
    logger.info("Starting batch processing")
    batch = BatchProcessor(
        base_dir,
        output_dir,
        max_workers=max_workers,
        skip_existing=skip_existing
    )
    stats = batch.run()
    if stats["failed"]:
        logger.error(
            f"Batch completed with failures: {len(stats['failed'])} paper(s)"
        )
        return 1
    logger.info("Batch processing completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())