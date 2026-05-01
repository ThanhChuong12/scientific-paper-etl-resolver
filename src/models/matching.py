"""
Reference Matching Pipeline
============================
This module implements a complete reference matching pipeline that matches
BibTeX entries (from refs.bib) with arXiv entries (from references.json).

Problem Type: RANKING/RETRIEVAL
- Input: A BibTeX entry (from refs.bib)
- Output: Ranked list of top 5 arXiv IDs from references.json
- Metric: Mean Reciprocal Rank (MRR@5)

Pipeline Steps:
1. Data Loading & Cleaning
"""

import json
import re
import os
import string
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================ CONSTANTS ================
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'we', 'you',
    'he', 'she', 'they', 'their', 'our', 'your', 'which', 'who', 'whom',
    'what', 'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'using',
    'via', 'towards', 'based', 'approach', 'method', 'methods', 'model',
    'models', 'learning', 'neural', 'network', 'networks', 'deep'
}


# ================ DATA CLASSES ================
@dataclass
class BibEntry:
    """Represents a BibTeX entry from refs.bib"""
    key: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: str = ""
    venue: str = ""  # journal/booktitle/etc.
    entry_type: str = ""
    doi: str = ""
    arxiv_id: str = ""  # If present in the bib entry
    raw_data: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.raw_data:
            self._extract_from_raw()
    
    def _extract_from_raw(self):
        """Extract fields from raw BibTeX data"""
        if not self.title:
            self.title = self.raw_data.get('title', '')
        if not self.authors:
            author_str = self.raw_data.get('author', '')
            self.authors = self._parse_authors(author_str)
        if not self.year:
            self.year = str(self.raw_data.get('year', ''))
        if not self.venue:
            self.venue = (self.raw_data.get('journal', '') or 
                         self.raw_data.get('booktitle', '') or
                         self.raw_data.get('publisher', ''))
        if not self.doi:
            self.doi = self.raw_data.get('doi', '')
        # Try to extract arxiv ID from journal field
        if not self.arxiv_id:
            journal = self.raw_data.get('journal', '').lower()
            if 'arxiv' in journal:
                match = re.search(r'(\d{4}\.\d{4,5})', journal)
                if match:
                    self.arxiv_id = match.group(1)
    
    def _parse_authors(self, author_str: str) -> List[str]:
        """Parse author string into list of authors"""
        if not author_str:
            return []
        # Handle "and" separator
        authors = re.split(r'\s+and\s+', author_str, flags=re.IGNORECASE)
        # Clean up each author
        cleaned = []
        for author in authors:
            author = author.strip()
            # Remove {Others} or similar
            if author.lower() in ['{others}', 'others', '{others}']:
                continue
            # Remove braces
            author = re.sub(r'[{}]', '', author)
            if author:
                cleaned.append(author)
        return cleaned


@dataclass  
class ArxivEntry:
    """Represents an arXiv entry from references.json"""
    arxiv_id: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: str = ""
    submission_date: str = ""
    semantic_scholar_id: str = ""
    
    @classmethod
    def from_dict(cls, arxiv_id: str, data: Dict) -> 'ArxivEntry':
        """Create ArxivEntry from dictionary"""
        title = data.get('title', data.get('paper_title', ''))
        authors = data.get('authors', [])
        submission_date = data.get('submission_date', '')
        year = ""
        if submission_date:
            match = re.search(r'(\d{4})', submission_date)
            if match:
                year = match.group(1)
        return cls(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            year=year,
            submission_date=submission_date,
            semantic_scholar_id=data.get('semantic_scholar_id', '')
        )


@dataclass
class MatchPair:
    """Represents a (BibEntry, ArxivEntry) pair with features and label"""
    publication_id: str
    bib_key: str
    arxiv_id: str
    features: Dict = field(default_factory=dict)
    label: int = -1  # -1: unlabeled, 0: no match, 1: match
    label_source: str = "none"  # manual, auto, none
    pred_score: float = 0.0
    rank: int = 0


# ================ TEXT PREPROCESSING ================
class TextCleaner:
    """Handles text cleaning and normalization"""
    
    # LaTeX commands to remove
    LATEX_PATTERNS = [
        (r'\\textbf\{([^}]*)\}', r'\1'),
        (r'\\textit\{([^}]*)\}', r'\1'),
        (r'\\emph\{([^}]*)\}', r'\1'),
        (r'\\textrm\{([^}]*)\}', r'\1'),
        (r'\\texttt\{([^}]*)\}', r'\1'),
        (r'\\mathrm\{([^}]*)\}', r'\1'),
        (r'\\mathbf\{([^}]*)\}', r'\1'),
        (r'\$([^$]*)\$', r'\1'),
        (r'\\[a-zA-Z]+\{([^}]*)\}', r'\1'),
        (r'[{}]', ''),
        (r'\\\\', ' '),
        (r'\\', ''),
    ]
    
    @classmethod
    def clean_text(cls, text: str, remove_stopwords: bool = False) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove LaTeX commands
        for pattern, replacement in cls.LATEX_PATTERNS:
            text = re.sub(pattern, replacement, text)
        
        # Lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = cls._normalize_unicode(text)
        
        # Remove punctuation except hyphens in words
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords if requested
        if remove_stopwords:
            words = text.split()
            words = [w for w in words if w not in STOPWORDS]
            text = ' '.join(words)
        
        return text
    
    @classmethod
    def _normalize_unicode(cls, text: str) -> str:
        """Normalize unicode characters to ASCII equivalents"""
        replacements = {
            'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss',
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
            'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
            'ñ': 'n', 'ç': 'c',
            '"': '"', '"': '"', ''': "'", ''': "'",
            '–': '-', '—': '-',
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text
    
    @classmethod
    def normalize_author_name(cls, author: str) -> str:
        """Normalize author name to 'lastname firstname' format"""
        if not author:
            return ""
        
        author = cls.clean_text(author)
        author = re.sub(r'[^\w\s-]', '', author)
        
        # Handle "Last, First" format
        if ',' in author:
            parts = author.split(',', 1)
            if len(parts) == 2:
                last = parts[0].strip()
                first = parts[1].strip()
                return f"{last} {first}".lower()
        
        # Handle "First Last" format - assume last word is lastname
        parts = author.split()
        if len(parts) >= 2:
            lastname = parts[-1]
            firstname = ' '.join(parts[:-1])
            return f"{lastname} {firstname}".lower()
        
        return author.lower()
    
    @classmethod
    def extract_lastname(cls, author: str) -> str:
        """Extract just the lastname from an author name"""
        if not author:
            return ""
        
        author = cls.clean_text(author)
        
        # Handle "Last, First" format
        if ',' in author:
            return author.split(',')[0].strip().lower()
        
        # Handle "First Last" format
        parts = author.split()
        if parts:
            return parts[-1].lower()
        
        return author.lower()


# ================ DATA LOADING ================
class DataLoader:
    """Handles loading and parsing of publication data"""
    
    def __init__(self, data_dir: Path, processed_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir) if processed_dir else None
    
    def load_publication(self, pub_id: str, use_processed: bool = False) -> Tuple[List[BibEntry], Dict[str, ArxivEntry]]:
        """
        Load BibTeX entries and arXiv references for a publication.
        
        Args:
            pub_id: Publication ID
            use_processed: If True, load from processed_dir instead of data_dir
        """
        if use_processed and self.processed_dir:
            pub_dir = self.processed_dir / pub_id
            # Processed folder has refs.bib directly in the folder
            bib_entries = self._load_bibtex(pub_dir / 'refs.bib')
        else:
            pub_dir = self.data_dir / pub_id
            # Data folder has .bib files nested in tex/ subfolder
            bib_entries = self._find_and_load_bibtex(pub_dir)
        
        # Load references.json
        arxiv_entries = self._load_references(pub_dir / 'references.json')
        
        return bib_entries, arxiv_entries
    
    def _find_and_load_bibtex(self, pub_dir: Path) -> List[BibEntry]:
        """Find and load .bib files from nested tex/ folder structure"""
        all_entries = []
        
        # First try direct refs.bib in folder
        if (pub_dir / 'refs.bib').exists():
            return self._load_bibtex(pub_dir / 'refs.bib')
        
        # Search in tex/ subfolder
        tex_dir = pub_dir / 'tex'
        if not tex_dir.exists():
            logger.warning(f"No tex/ folder found in {pub_dir}")
            return []
        
        # Find all .bib files recursively
        bib_files = list(tex_dir.rglob('*.bib'))
        
        if not bib_files:
            logger.warning(f"No .bib files found in {tex_dir}")
            return []
        
        # Sort to get latest version (e.g., v2 over v1)
        bib_files = sorted(bib_files, key=lambda x: str(x), reverse=True)
        
        # Load entries from all bib files, prioritizing latest versions
        seen_keys = set()
        for bib_path in bib_files:
            entries = self._load_bibtex(bib_path)
            for entry in entries:
                if entry.key not in seen_keys:
                    all_entries.append(entry)
                    seen_keys.add(entry.key)
        
        logger.info(f"Loaded {len(all_entries)} unique BibTeX entries from {len(bib_files)} files in {pub_dir.name}")
        return all_entries
    
    def _load_bibtex(self, bib_path: Path) -> List[BibEntry]:
        """Parse BibTeX file into list of BibEntry objects"""
        if not bib_path.exists():
            logger.warning(f"BibTeX file not found: {bib_path}")
            return []
        
        try:
            content = bib_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = bib_path.read_text(encoding='latin-1')
        
        entries = []
        
        # Pattern to match BibTeX entries
        entry_pattern = re.compile(
            r'@(\w+)\s*\{\s*([^,\s]+)\s*,\s*(.*?)\n\s*\}',
            re.DOTALL
        )
        
        for match in entry_pattern.finditer(content):
            entry_type = match.group(1)
            key = match.group(2)
            fields_str = match.group(3)
            
            # Parse fields
            fields = self._parse_bib_fields(fields_str)
            fields['entry_type'] = entry_type
            
            entry = BibEntry(
                key=key,
                entry_type=entry_type,
                raw_data=fields
            )
            entries.append(entry)
        
        logger.info(f"Loaded {len(entries)} BibTeX entries from {bib_path.name}")
        return entries
    
    def _parse_bib_fields(self, fields_str: str) -> Dict[str, str]:
        """Parse BibTeX fields string into dictionary"""
        fields = {}
        
        # Pattern to match field = value pairs
        field_pattern = re.compile(
            r'(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|"([^"]*)"|\s*(\d+))',
            re.DOTALL
        )
        
        for match in field_pattern.finditer(fields_str):
            field_name = match.group(1).lower()
            # Get value from either braces, quotes, or bare number
            value = match.group(2) or match.group(3) or match.group(4) or ""
            value = value.strip()
            fields[field_name] = value
        
        return fields
    
    def _load_references(self, ref_path: Path) -> Dict[str, ArxivEntry]:
        """Load references.json into dictionary of ArxivEntry objects"""
        if not ref_path.exists():
            logger.warning(f"References file not found: {ref_path}")
            return {}
        
        try:
            with open(ref_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing references.json: {e}")
            return {}
        
        if not data:
            logger.warning(f"Empty references.json: {ref_path}")
            return {}
        
        entries = {}
        for arxiv_id, entry_data in data.items():
            entries[arxiv_id] = ArxivEntry.from_dict(arxiv_id, entry_data)
        
        logger.info(f"Loaded {len(entries)} arXiv entries from {ref_path.name}")
        return entries
    
    def get_all_publication_ids(self, use_processed: bool = False) -> List[str]:
        """Get list of all publication IDs in data directory"""
        target_dir = self.processed_dir if use_processed and self.processed_dir else self.data_dir
        pub_ids = []
        for item in target_dir.iterdir():
            if item.is_dir():
                # Check if has refs.bib or any .bib file in tex/ subfolder
                has_refs = (item / 'refs.bib').exists()
                has_tex_bib = any((item / 'tex').rglob('*.bib')) if (item / 'tex').exists() else False
                has_references = (item / 'references.json').exists()
                
                if (has_refs or has_tex_bib) and has_references:
                    pub_ids.append(item.name)
        return sorted(pub_ids)
    
    def load_manual_labels(self, pub_id: str) -> Dict[str, str]:
        """Load manual labels from labels_manual.json if exists"""
        if self.processed_dir:
            labels_path = self.processed_dir / pub_id / 'labels_manual.json'
        else:
            labels_path = self.data_dir / pub_id / 'labels_manual.json'
        
        if labels_path.exists():
            with open(labels_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
