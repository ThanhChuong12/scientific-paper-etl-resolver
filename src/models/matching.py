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
2. Data Labeling (Manual + Automatic)
3. Feature Engineering
4. Model Training (XGBoost Ranker)
5. Evaluation (MRR@5)x
6. Prediction Generation (pred.json)
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


# ================ FEATURE ENGINEERING ================
class FeatureExtractor:
    """Extracts features for matching pairs"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self._fitted = False
    
    def fit(self, titles: List[str]):
        """Fit TF-IDF vectorizer on corpus of titles"""
        cleaned_titles = [TextCleaner.clean_text(t) for t in titles]
        self.tfidf_vectorizer.fit(cleaned_titles)
        self._fitted = True
    
    def extract_features(self, bib: BibEntry, arxiv: ArxivEntry) -> Dict[str, float]:
        """Extract all features for a (BibTeX, ArxivEntry) pair"""
        features = {}
        
        # 1. Title similarity features
        title_features = self._title_features(bib.title, arxiv.title)
        features.update(title_features)
        
        # 2. Author similarity features
        author_features = self._author_features(bib.authors, arxiv.authors)
        features.update(author_features)
        
        # 3. Year features
        year_features = self._year_features(bib.year, arxiv.year)
        features.update(year_features)
        
        # 4. Venue/arXiv features
        venue_features = self._venue_features(bib.venue, arxiv.arxiv_id)
        features.update(venue_features)
        
        # 5. Key-based features (arXiv ID in bib key)
        key_features = self._key_features(bib.key, bib.arxiv_id, arxiv.arxiv_id)
        features.update(key_features)
        
        return features
    
    def _title_features(self, bib_title: str, arxiv_title: str) -> Dict[str, float]:
        """Extract title similarity features"""
        bib_clean = TextCleaner.clean_text(bib_title)
        arxiv_clean = TextCleaner.clean_text(arxiv_title)
        
        bib_no_stop = TextCleaner.clean_text(bib_title, remove_stopwords=True)
        arxiv_no_stop = TextCleaner.clean_text(arxiv_title, remove_stopwords=True)
        
        features = {}
        
        # 1. Exact match
        features['title_exact_match'] = float(bib_clean == arxiv_clean)
        
        # 2. Jaccard similarity on tokens
        bib_tokens = set(bib_clean.split())
        arxiv_tokens = set(arxiv_clean.split())
        if bib_tokens or arxiv_tokens:
            features['title_jaccard'] = len(bib_tokens & arxiv_tokens) / len(bib_tokens | arxiv_tokens)
        else:
            features['title_jaccard'] = 0.0
        
        # 3. Jaccard without stopwords
        bib_tokens_ns = set(bib_no_stop.split())
        arxiv_tokens_ns = set(arxiv_no_stop.split())
        if bib_tokens_ns or arxiv_tokens_ns:
            features['title_jaccard_nostop'] = len(bib_tokens_ns & arxiv_tokens_ns) / len(bib_tokens_ns | arxiv_tokens_ns)
        else:
            features['title_jaccard_nostop'] = 0.0
        
        # 4. TF-IDF cosine similarity
        if self._fitted:
            try:
                tfidf_matrix = self.tfidf_vectorizer.transform([bib_clean, arxiv_clean])
                features['title_tfidf_cosine'] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                features['title_tfidf_cosine'] = 0.0
        else:
            features['title_tfidf_cosine'] = 0.0
        
        # 5. Sequence matcher ratio (Levenshtein-like)
        features['title_sequence_ratio'] = SequenceMatcher(None, bib_clean, arxiv_clean).ratio()
        
        # 6. Length ratio
        if max(len(bib_clean), len(arxiv_clean)) > 0:
            features['title_len_ratio'] = min(len(bib_clean), len(arxiv_clean)) / max(len(bib_clean), len(arxiv_clean))
        else:
            features['title_len_ratio'] = 0.0
        
        # 7. Word count ratio
        bib_words = len(bib_clean.split())
        arxiv_words = len(arxiv_clean.split())
        if max(bib_words, arxiv_words) > 0:
            features['title_word_ratio'] = min(bib_words, arxiv_words) / max(bib_words, arxiv_words)
        else:
            features['title_word_ratio'] = 0.0
        
        # 8. Common bigrams
        bib_bigrams = self._get_ngrams(bib_no_stop, 2)
        arxiv_bigrams = self._get_ngrams(arxiv_no_stop, 2)
        if bib_bigrams or arxiv_bigrams:
            features['title_bigram_overlap'] = len(bib_bigrams & arxiv_bigrams) / max(1, len(bib_bigrams | arxiv_bigrams))
        else:
            features['title_bigram_overlap'] = 0.0
        
        # 9. Common trigrams  
        bib_trigrams = self._get_ngrams(bib_no_stop, 3)
        arxiv_trigrams = self._get_ngrams(arxiv_no_stop, 3)
        if bib_trigrams or arxiv_trigrams:
            features['title_trigram_overlap'] = len(bib_trigrams & arxiv_trigrams) / max(1, len(bib_trigrams | arxiv_trigrams))
        else:
            features['title_trigram_overlap'] = 0.0
        
        # 10. Contains key terms
        key_terms = ['survey', 'review', 'benchmark', 'analysis', 'study', 'evaluation']
        bib_has_key = any(term in bib_clean for term in key_terms)
        arxiv_has_key = any(term in arxiv_clean for term in key_terms)
        features['title_keyterm_match'] = float(bib_has_key == arxiv_has_key)
        
        return features
    
    def _author_features(self, bib_authors: List[str], arxiv_authors: List[str]) -> Dict[str, float]:
        """Extract author similarity features"""
        features = {}
        
        if not bib_authors or not arxiv_authors:
            features['author_jaccard'] = 0.0
            features['author_first_match'] = 0.0
            features['author_last_match'] = 0.0
            features['author_count_ratio'] = 0.0
            features['author_lastname_overlap'] = 0.0
            features['author_any_match'] = 0.0
            return features
        
        # Normalize author names
        bib_normalized = [TextCleaner.normalize_author_name(a) for a in bib_authors]
        arxiv_normalized = [TextCleaner.normalize_author_name(a) for a in arxiv_authors]
        
        # Extract lastnames
        bib_lastnames = set(TextCleaner.extract_lastname(a) for a in bib_authors)
        arxiv_lastnames = set(TextCleaner.extract_lastname(a) for a in arxiv_authors)
        
        # 1. Full name Jaccard
        bib_set = set(bib_normalized)
        arxiv_set = set(arxiv_normalized)
        features['author_jaccard'] = len(bib_set & arxiv_set) / max(1, len(bib_set | arxiv_set))
        
        # 2. First author match (lastname)
        bib_first_last = TextCleaner.extract_lastname(bib_authors[0]) if bib_authors else ""
        arxiv_first_last = TextCleaner.extract_lastname(arxiv_authors[0]) if arxiv_authors else ""
        features['author_first_match'] = float(bib_first_last == arxiv_first_last and bib_first_last != "")
        
        # 3. Last author match (lastname)
        bib_last_last = TextCleaner.extract_lastname(bib_authors[-1]) if bib_authors else ""
        arxiv_last_last = TextCleaner.extract_lastname(arxiv_authors[-1]) if arxiv_authors else ""
        features['author_last_match'] = float(bib_last_last == arxiv_last_last and bib_last_last != "")
        
        # 4. Author count ratio
        features['author_count_ratio'] = min(len(bib_authors), len(arxiv_authors)) / max(len(bib_authors), len(arxiv_authors))
        
        # 5. Lastname overlap
        if bib_lastnames or arxiv_lastnames:
            features['author_lastname_overlap'] = len(bib_lastnames & arxiv_lastnames) / max(1, len(bib_lastnames | arxiv_lastnames))
        else:
            features['author_lastname_overlap'] = 0.0
        
        # 6. Any author match
        features['author_any_match'] = float(len(bib_lastnames & arxiv_lastnames) > 0)
        
        return features
    
    def _year_features(self, bib_year: str, arxiv_year: str) -> Dict[str, float]:
        """Extract year-based features"""
        features = {}
        
        try:
            bib_y = int(bib_year) if bib_year and bib_year.isdigit() else 0
            arxiv_y = int(arxiv_year) if arxiv_year and arxiv_year.isdigit() else 0
        except:
            bib_y, arxiv_y = 0, 0
        
        # 1. Exact year match
        features['year_exact_match'] = float(bib_y == arxiv_y and bib_y > 0)
        
        # 2. Year difference
        if bib_y > 0 and arxiv_y > 0:
            features['year_diff'] = abs(bib_y - arxiv_y)
            features['year_diff_normalized'] = 1.0 / (1.0 + abs(bib_y - arxiv_y))
        else:
            features['year_diff'] = 10  # Default large difference
            features['year_diff_normalized'] = 0.1
        
        # 3. Year within range (arXiv typically precedes publication)
        if bib_y > 0 and arxiv_y > 0:
            diff = bib_y - arxiv_y
            features['year_arxiv_before_pub'] = float(0 <= diff <= 2)
        else:
            features['year_arxiv_before_pub'] = 0.0
        
        return features
    
    def _venue_features(self, venue: str, arxiv_id: str) -> Dict[str, float]:
        """Extract venue/arXiv related features"""
        features = {}
        
        venue_clean = venue.lower() if venue else ""
        
        # 1. Is arXiv preprint
        features['venue_is_arxiv'] = float('arxiv' in venue_clean or 'preprint' in venue_clean)
        
        # 2. ArXiv ID format in venue
        arxiv_pattern = re.search(r'(\d{4}\.\d{4,5})', venue_clean)
        features['venue_has_arxiv_id'] = float(arxiv_pattern is not None)
        
        # 3. ArXiv ID match
        if arxiv_pattern:
            venue_arxiv = arxiv_pattern.group(1)
            arxiv_normalized = arxiv_id.replace('-', '.')
            features['venue_arxiv_id_match'] = float(venue_arxiv == arxiv_normalized)
        else:
            features['venue_arxiv_id_match'] = 0.0
        
        return features
    
    def _key_features(self, bib_key: str, bib_arxiv_id: str, arxiv_id: str) -> Dict[str, float]:
        """Extract features based on BibTeX key"""
        features = {}
        
        bib_key_lower = bib_key.lower()
        arxiv_normalized = arxiv_id.replace('-', '').replace('.', '')
        
        # 1. ArXiv ID in bib key
        features['key_has_arxiv'] = float(arxiv_normalized in bib_key_lower.replace('-', '').replace('.', ''))
        
        # 2. Extracted arXiv ID match
        if bib_arxiv_id:
            bib_arxiv_norm = bib_arxiv_id.replace('-', '.').replace('_', '.')
            arxiv_norm = arxiv_id.replace('-', '.')
            features['bib_arxiv_match'] = float(bib_arxiv_norm == arxiv_norm)
        else:
            features['bib_arxiv_match'] = 0.0
        
        # 3. Year in key matches
        key_year_match = re.search(r'(\d{4})', bib_key)
        if key_year_match:
            key_year = key_year_match.group(1)
            # Check if arxiv_id starts with year pattern (e.g., 2412- means 2024)
            if arxiv_id[:2].isdigit():
                arxiv_year_prefix = arxiv_id[:2]
                # 24 -> 2024, 23 -> 2023, etc.
                arxiv_year = f"20{arxiv_year_prefix}"
                features['key_year_match'] = float(key_year == arxiv_year)
            else:
                features['key_year_match'] = 0.0
        else:
            features['key_year_match'] = 0.0
        
        return features
    
    def _get_ngrams(self, text: str, n: int) -> Set[str]:
        """Get n-grams from text"""
        words = text.split()
        if len(words) < n:
            return set()
        return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))


# ================ DATA LABELING ================
class DataLabeler:
    """Handles manual and automatic data labeling"""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
    
    def create_manual_labels(
        self, 
        pub_id: str,
        bib_entries: List[BibEntry],
        arxiv_entries: Dict[str, ArxivEntry],
        known_matches: Dict[str, str]  # bib_key -> arxiv_id
    ) -> List[MatchPair]:
        """
        Create labeled pairs for manual labeling.
        known_matches: Dictionary mapping bib_key to correct arxiv_id
        """
        pairs = []
        
        for bib in bib_entries:
            for arxiv_id, arxiv in arxiv_entries.items():
                # Check if this is a known match
                is_match = known_matches.get(bib.key) == arxiv_id
                
                features = self.feature_extractor.extract_features(bib, arxiv)
                
                pair = MatchPair(
                    publication_id=pub_id,
                    bib_key=bib.key,
                    arxiv_id=arxiv_id,
                    features=features,
                    label=1 if is_match else 0,
                    label_source="manual"
                )
                pairs.append(pair)
        
        return pairs
    
    def create_auto_labels(
        self,
        pub_id: str,
        bib_entries: List[BibEntry],
        arxiv_entries: Dict[str, ArxivEntry],
        threshold: float = 0.7
    ) -> Tuple[List[MatchPair], Dict[str, str]]:
        """
        Create automatically labeled pairs using heuristics.
        Returns pairs and discovered matches.
        """
        pairs = []
        discovered_matches = {}
        
        for bib in bib_entries:
            best_score = 0.0
            best_arxiv_id = None
            
            for arxiv_id, arxiv in arxiv_entries.items():
                features = self.feature_extractor.extract_features(bib, arxiv)
                
                # Calculate heuristic score
                score = self._calculate_heuristic_score(features)
                
                if score > best_score:
                    best_score = score
                    best_arxiv_id = arxiv_id
                
                # Determine label
                is_match = score >= threshold
                
                pair = MatchPair(
                    publication_id=pub_id,
                    bib_key=bib.key,
                    arxiv_id=arxiv_id,
                    features=features,
                    label=1 if is_match else 0,
                    label_source="auto"
                )
                pairs.append(pair)
            
            # Record best match if above threshold
            if best_score >= threshold and best_arxiv_id:
                discovered_matches[bib.key] = best_arxiv_id
        
        return pairs, discovered_matches
    
    def _calculate_heuristic_score(self, features: Dict[str, float]) -> float:
        """Calculate matching score using heuristics"""
        score = 0.0
        
        # Strong indicators (high weight)
        if features.get('title_exact_match', 0) > 0:
            return 1.0
        if features.get('venue_arxiv_id_match', 0) > 0:
            return 0.95
        if features.get('bib_arxiv_match', 0) > 0:
            return 0.95
        if features.get('key_has_arxiv', 0) > 0:
            score += 0.3
        
        # Title similarity (high weight)
        title_jaccard = features.get('title_jaccard_nostop', 0)
        score += title_jaccard * 0.35
        
        title_sequence = features.get('title_sequence_ratio', 0)
        score += title_sequence * 0.2
        
        # Author similarity (medium weight)
        author_overlap = features.get('author_lastname_overlap', 0)
        score += author_overlap * 0.2
        
        first_author = features.get('author_first_match', 0)
        score += first_author * 0.1
        
        # Year match (low weight but important)
        year_match = features.get('year_exact_match', 0)
        score += year_match * 0.1
        
        year_norm = features.get('year_diff_normalized', 0)
        score += year_norm * 0.05
        
        return min(score, 1.0)
