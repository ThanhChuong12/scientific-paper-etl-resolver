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


# ================ MODEL TRAINING ================
class ReferenceMatcherModel:
    """Machine learning model for reference matching"""
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the underlying ML model"""
        if self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, pairs: List[MatchPair], valid_pairs: Optional[List[MatchPair]] = None):
        """Train the model on labeled pairs"""
        # Convert to DataFrame
        df = self._pairs_to_dataframe(pairs)
        
        # Get feature columns
        self.feature_cols = [col for col in df.columns 
                           if col not in ['publication_id', 'bib_key', 'arxiv_id', 'label', 'label_source']]
        
        X = df[self.feature_cols].values
        y = df['label'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        logger.info(f"Model trained on {len(pairs)} pairs")
        
        # Evaluate on validation if provided
        if valid_pairs:
            valid_df = self._pairs_to_dataframe(valid_pairs)
            X_valid = valid_df[self.feature_cols].values
            y_valid = valid_df['label'].values
            X_valid_scaled = self.scaler.transform(X_valid)
            
            valid_acc = self.model.score(X_valid_scaled, y_valid)
            logger.info(f"Validation accuracy: {valid_acc:.4f}")
    
    def predict_scores(self, pairs: List[MatchPair]) -> List[MatchPair]:
        """Predict matching scores for pairs"""
        if not self.feature_cols:
            raise ValueError("Model not trained yet")
        
        df = self._pairs_to_dataframe(pairs)
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Get probability of positive class
        scores = self.model.predict_proba(X_scaled)[:, 1]
        
        # Update pairs with scores
        for pair, score in zip(pairs, scores):
            pair.pred_score = score
        
        return pairs
    
    def rank_candidates(self, pairs: List[MatchPair]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Rank candidates for each BibTeX entry.
        Returns: {bib_key: [(arxiv_id, score), ...]}
        """
        # Predict scores
        pairs = self.predict_scores(pairs)
        
        # Group by bib_key
        by_bib_key = defaultdict(list)
        for pair in pairs:
            by_bib_key[pair.bib_key].append((pair.arxiv_id, pair.pred_score))
        
        # Sort by score (descending)
        ranked = {}
        for bib_key, candidates in by_bib_key.items():
            ranked[bib_key] = sorted(candidates, key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def _pairs_to_dataframe(self, pairs: List[MatchPair]) -> pd.DataFrame:
        """Convert list of MatchPair to DataFrame"""
        records = []
        for pair in pairs:
            record = {
                'publication_id': pair.publication_id,
                'bib_key': pair.bib_key,
                'arxiv_id': pair.arxiv_id,
                'label': pair.label,
                'label_source': pair.label_source,
                **pair.features
            }
            records.append(record)
        return pd.DataFrame(records)


# ================ EVALUATION ================
class Evaluator:
    """Evaluates matching performance using MRR@K"""
    
    @staticmethod
    def calculate_mrr(
        ranked_candidates: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, str],
        k: int = 5
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Mean Reciprocal Rank @ K
        
        Args:
            ranked_candidates: {bib_key: [(arxiv_id, score), ...]}
            ground_truth: {bib_key: correct_arxiv_id}
            k: Top-k candidates to consider
            
        Returns:
            (mrr_score, {bib_key: reciprocal_rank})
        """
        reciprocal_ranks = {}
        
        for bib_key, correct_arxiv in ground_truth.items():
            if bib_key not in ranked_candidates:
                reciprocal_ranks[bib_key] = 0.0
                continue
            
            # Get top-k candidates
            top_k = ranked_candidates[bib_key][:k]
            top_k_ids = [arxiv_id for arxiv_id, _ in top_k]
            
            # Find rank of correct answer
            if correct_arxiv in top_k_ids:
                rank = top_k_ids.index(correct_arxiv) + 1
                reciprocal_ranks[bib_key] = 1.0 / rank
            else:
                reciprocal_ranks[bib_key] = 0.0
        
        # Calculate MRR
        if reciprocal_ranks:
            mrr = np.mean(list(reciprocal_ranks.values()))
        else:
            mrr = 0.0
        
        return mrr, reciprocal_ranks
    
    @staticmethod
    def generate_predictions(
        ranked_candidates: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, str],
        partition: str,
        k: int = 5
    ) -> Dict:
        """Generate predictions in required format"""
        predictions = {}
        
        for bib_key in ranked_candidates:
            top_k = ranked_candidates[bib_key][:k]
            predictions[bib_key] = [arxiv_id for arxiv_id, _ in top_k]
        
        return {
            "partition": partition,
            "groundtruth": ground_truth,
            "prediction": predictions
        }


# ================ MAIN PIPELINE ================
class ReferenceMatchingPipeline:
    """Complete reference matching pipeline"""
    
    def __init__(self, data_dir: Path, processed_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.output_dir = output_dir or self.data_dir
        
        self.data_loader = DataLoader(data_dir, processed_dir)
        self.feature_extractor = FeatureExtractor()
        self.labeler = DataLabeler(self.feature_extractor)
        self.model = ReferenceMatcherModel(model_type='gradient_boosting')
        self.evaluator = Evaluator()
        
        # Data storage
        self.publications = {}  # {pub_id: (bib_entries, arxiv_entries)}
        self.all_pairs = {}  # {pub_id: List[MatchPair]}
        self.ground_truth = {}  # {pub_id: {bib_key: arxiv_id}}
        self.split_info = {}  # {pub_id: 'train'/'valid'/'test'}
        self.label_config = {}  # Store label.json configuration
        
    def load_label_config(self, label_json_path: Path) -> Dict:
        """Load label.json configuration file"""
        with open(label_json_path, 'r', encoding='utf-8') as f:
            self.label_config = json.load(f)
        logger.info(f"Loaded label config: {self.label_config['manual_subset']['count']} manual, {self.label_config['auto_subset']['count']} auto")
        return self.label_config
    
    def load_all_publications(self, manual_only: bool = False, auto_only: bool = False, max_auto: int = None) -> int:
        """
        Load all publications from data directory.
        
        Args:
            manual_only: Load only manually labeled publications
            auto_only: Load only auto-labeled publications
            max_auto: Maximum number of auto publications to load (for faster testing)
        """
        manual_pubs = set(self.label_config.get('manual_subset', {}).get('papers', []))
        auto_pubs = self.label_config.get('auto_subset', {}).get('papers', [])
        
        if max_auto:
            auto_pubs = auto_pubs[:max_auto]
        
        loaded_count = 0
        
        # Load manual publications from processed_dir
        if not auto_only and self.processed_dir:
            for pub_id in manual_pubs:
                try:
                    bib_entries, arxiv_entries = self.data_loader.load_publication(pub_id, use_processed=True)
                    if bib_entries and arxiv_entries:
                        self.publications[pub_id] = (bib_entries, arxiv_entries, 'manual')
                        loaded_count += 1
                        logger.info(f"[Manual] Loaded {pub_id}: {len(bib_entries)} bib, {len(arxiv_entries)} arxiv")
                except Exception as e:
                    logger.error(f"Failed to load manual publication {pub_id}: {e}")
        
        # Load auto publications from data_dir
        if not manual_only:
            for pub_id in auto_pubs:
                try:
                    bib_entries, arxiv_entries = self.data_loader.load_publication(pub_id, use_processed=False)
                    if bib_entries and arxiv_entries:
                        self.publications[pub_id] = (bib_entries, arxiv_entries, 'auto')
                        loaded_count += 1
                        if loaded_count % 50 == 0:
                            logger.info(f"Loaded {loaded_count} publications...")
                except Exception as e:
                    logger.warning(f"Failed to load auto publication {pub_id}: {e}")
        
        logger.info(f"Total loaded: {loaded_count} publications")
        return loaded_count
    
    def fit_feature_extractor(self):
        """Fit TF-IDF vectorizer on all titles"""
        all_titles = []
        for pub_id, pub_data in self.publications.items():
            bib_entries, arxiv_entries, _ = pub_data
            for bib in bib_entries:
                all_titles.append(bib.title)
            for arxiv in arxiv_entries.values():
                all_titles.append(arxiv.title)
        
        self.feature_extractor.fit(all_titles)
        logger.info(f"Feature extractor fitted on {len(all_titles)} titles")
    
    def create_labels(
        self,
        manual_labels: Optional[Dict[str, Dict[str, str]]] = None,
        auto_threshold: float = 0.7,
        load_manual_from_files: bool = True
    ):
        """
        Create labels for all publications.
        
        Args:
            manual_labels: Manually labeled ground truth for specific publications
            auto_threshold: Threshold for automatic labeling
            load_manual_from_files: If True, load manual labels from labels_manual.json files
        """
        manual_labels = manual_labels or {}
        
        # Load manual labels from files if requested
        if load_manual_from_files and self.processed_dir:
            for pub_id, pub_data in self.publications.items():
                if pub_data[2] == 'manual':  # This is a manual publication
                    file_labels = self.data_loader.load_manual_labels(pub_id)
                    if file_labels:
                        manual_labels[pub_id] = file_labels
                        logger.info(f"Loaded {len(file_labels)} manual labels for {pub_id}")
        
        manual_pub_ids = set(manual_labels.keys())
        
        for pub_id, pub_data in self.publications.items():
            bib_entries, arxiv_entries, label_type = pub_data
            
            if pub_id in manual_pub_ids:
                # Use manual labels
                known_matches = manual_labels[pub_id]
                pairs = self.labeler.create_manual_labels(
                    pub_id, bib_entries, arxiv_entries, known_matches
                )
                self.ground_truth[pub_id] = known_matches
                logger.info(f"Manual labels for {pub_id}: {len(known_matches)} matches")
            else:
                # Use automatic labels
                pairs, discovered = self.labeler.create_auto_labels(
                    pub_id, bib_entries, arxiv_entries, threshold=auto_threshold
                )
                self.ground_truth[pub_id] = discovered
                if discovered:
                    logger.info(f"Auto labels for {pub_id}: {len(discovered)} matches discovered")
            
            self.all_pairs[pub_id] = pairs
    
    def split_data(
        self,
        test_pubs: List[str],
        valid_pubs: List[str]
    ):
        """
        Split publications into train/valid/test sets.
        
        According to requirements:
        - Test Set: 1 publication from manual + 1 from auto
        - Validation Set: 1 publication from manual + 1 from auto
        - Training Set: All remaining publications
        """
        test_set = set(test_pubs)
        valid_set = set(valid_pubs)
        
        for pub_id in self.publications:
            if pub_id in test_set:
                self.split_info[pub_id] = 'test'
            elif pub_id in valid_set:
                self.split_info[pub_id] = 'valid'
            else:
                self.split_info[pub_id] = 'train'
        
        train_count = sum(1 for v in self.split_info.values() if v == 'train')
        valid_count = sum(1 for v in self.split_info.values() if v == 'valid')
        test_count = sum(1 for v in self.split_info.values() if v == 'test')
        
        logger.info(f"Data split: train={train_count}, valid={valid_count}, test={test_count}")
        
        return {
            'train': train_count,
            'valid': valid_count,
            'test': test_count
        }
    
    def auto_split_data(self):
        """
        Automatically split data according to requirements:
        - Test: 1 manual + 1 auto
        - Valid: 1 manual + 1 auto
        - Train: remaining
        """
        manual_pubs = [pub_id for pub_id, data in self.publications.items() if data[2] == 'manual']
        auto_pubs = [pub_id for pub_id, data in self.publications.items() if data[2] == 'auto']
        
        # Select publications for test and validation
        test_manual = manual_pubs[0] if len(manual_pubs) > 0 else None
        test_auto = auto_pubs[0] if len(auto_pubs) > 0 else None
        valid_manual = manual_pubs[1] if len(manual_pubs) > 1 else None
        valid_auto = auto_pubs[1] if len(auto_pubs) > 1 else None
        
        test_pubs = [p for p in [test_manual, test_auto] if p]
        valid_pubs = [p for p in [valid_manual, valid_auto] if p]
        
        logger.info(f"Auto-split: test={test_pubs}, valid={valid_pubs}")
        
        return self.split_data(test_pubs, valid_pubs)
    
    def get_pairs_by_split(self, split: str) -> List[MatchPair]:
        """Get all pairs for a given split"""
        pairs = []
        for pub_id, pub_pairs in self.all_pairs.items():
            if self.split_info.get(pub_id) == split:
                pairs.extend(pub_pairs)
        return pairs
    
    def train_model(self):
        """Train the matching model"""
        train_pairs = self.get_pairs_by_split('train')
        valid_pairs = self.get_pairs_by_split('valid')
        
        if not train_pairs:
            logger.warning("No training pairs available")
            return
        
        self.model.train(train_pairs, valid_pairs)
    
    def evaluate(self, split: str = 'test', k: int = 5) -> Tuple[float, Dict]:
        """Evaluate model on a split"""
        results = {}
        all_mrr_scores = []
        
        for pub_id, pub_pairs in self.all_pairs.items():
            if self.split_info.get(pub_id) != split:
                continue
            
            # Get ranked candidates
            ranked = self.model.rank_candidates(pub_pairs)
            
            # Get ground truth for this publication
            gt = self.ground_truth.get(pub_id, {})
            
            # Calculate MRR
            mrr, per_entry_rr = self.evaluator.calculate_mrr(ranked, gt, k=k)
            
            results[pub_id] = {
                'mrr': mrr,
                'per_entry': per_entry_rr,
                'ranked_candidates': ranked,
                'ground_truth': gt
            }
            
            all_mrr_scores.append(mrr)
            logger.info(f"MRR@{k} for {pub_id}: {mrr:.4f}")
        
        overall_mrr = np.mean(all_mrr_scores) if all_mrr_scores else 0.0
        logger.info(f"Overall MRR@{k} on {split}: {overall_mrr:.4f}")
        
        return overall_mrr, results
    
    def generate_and_save_predictions(self, k: int = 5):
        """Generate predictions and save to pred.json files"""
        for pub_id, pub_pairs in self.all_pairs.items():
            split = self.split_info.get(pub_id, 'train')
            
            # Get ranked candidates
            ranked = self.model.rank_candidates(pub_pairs)
            
            # Get ground truth
            gt = self.ground_truth.get(pub_id, {})
            
            # Generate predictions
            pred_data = self.evaluator.generate_predictions(ranked, gt, split, k=k)
            
            # Determine output path based on publication type
            if pub_id in [p for p, d in self.publications.items() if d[2] == 'manual']:
                pred_path = self.processed_dir / pub_id / 'pred.json' if self.processed_dir else self.data_dir / pub_id / 'pred.json'
            else:
                pred_path = self.data_dir / pub_id / 'pred.json'
            
            # Ensure directory exists
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(pred_path, 'w', encoding='utf-8') as f:
                json.dump(pred_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved predictions to {pred_path}")
    
    def run_full_pipeline(
        self,
        manual_labels: Dict[str, Dict[str, str]],
        test_pubs: List[str],
        valid_pubs: List[str],
        auto_threshold: float = 0.7,
        k: int = 5
    ) -> Dict:
        """Run the complete pipeline"""
        # 1. Load data
        logger.info("=" * 50)
        logger.info("Step 1: Loading publications...")
        n_loaded = self.load_all_publications()
        logger.info(f"Loaded {n_loaded} publications")
        
        if n_loaded == 0:
            logger.error("No publications loaded!")
            return {}
        
        # 2. Fit feature extractor
        logger.info("=" * 50)
        logger.info("Step 2: Fitting feature extractor...")
        self.fit_feature_extractor()
        
        # 3. Create labels
        logger.info("=" * 50)
        logger.info("Step 3: Creating labels...")
        self.create_labels(manual_labels, auto_threshold)
        
        # 4. Split data
        logger.info("=" * 50)
        logger.info("Step 4: Splitting data...")
        self.split_data(test_pubs, valid_pubs)
        
        # 5. Train model
        logger.info("=" * 50)
        logger.info("Step 5: Training model...")
        self.train_model()
        
        # 6. Evaluate
        logger.info("=" * 50)
        logger.info("Step 6: Evaluating on test set...")
        test_mrr, test_results = self.evaluate('test', k=k)
        
        logger.info("Evaluating on validation set...")
        valid_mrr, valid_results = self.evaluate('valid', k=k)
        
        # 7. Generate predictions
        logger.info("=" * 50)
        logger.info("Step 7: Generating predictions...")
        self.generate_and_save_predictions(k=k)
        
        return {
            'test_mrr': test_mrr,
            'valid_mrr': valid_mrr,
            'test_results': test_results,
            'valid_results': valid_results
        }


# ================ EDA UTILITIES ================
class EDAUtils:
    """Utilities for Exploratory Data Analysis"""
    
    @staticmethod
    def analyze_publication(
        bib_entries: List[BibEntry],
        arxiv_entries: Dict[str, ArxivEntry]
    ) -> Dict:
        """Analyze a single publication's data"""
        stats = {
            'n_bib': len(bib_entries),
            'n_arxiv': len(arxiv_entries),
            'n_possible_pairs': len(bib_entries) * len(arxiv_entries),
        }
        
        # BibTeX statistics
        bib_title_lens = [len(b.title.split()) for b in bib_entries]
        bib_author_counts = [len(b.authors) for b in bib_entries]
        bib_years = [int(b.year) for b in bib_entries if b.year.isdigit()]
        
        stats['bib_title_len_mean'] = np.mean(bib_title_lens) if bib_title_lens else 0
        stats['bib_title_len_std'] = np.std(bib_title_lens) if bib_title_lens else 0
        stats['bib_author_count_mean'] = np.mean(bib_author_counts) if bib_author_counts else 0
        stats['bib_year_range'] = (min(bib_years), max(bib_years)) if bib_years else (0, 0)
        
        # ArXiv statistics
        arxiv_title_lens = [len(a.title.split()) for a in arxiv_entries.values()]
        arxiv_author_counts = [len(a.authors) for a in arxiv_entries.values()]
        arxiv_years = [int(a.year) for a in arxiv_entries.values() if a.year.isdigit()]
        
        stats['arxiv_title_len_mean'] = np.mean(arxiv_title_lens) if arxiv_title_lens else 0
        stats['arxiv_title_len_std'] = np.std(arxiv_title_lens) if arxiv_title_lens else 0
        stats['arxiv_author_count_mean'] = np.mean(arxiv_author_counts) if arxiv_author_counts else 0
        stats['arxiv_year_range'] = (min(arxiv_years), max(arxiv_years)) if arxiv_years else (0, 0)
        
        return stats
    
    @staticmethod
    def find_potential_matches_heuristic(
        bib_entries: List[BibEntry],
        arxiv_entries: Dict[str, ArxivEntry],
        feature_extractor: FeatureExtractor,
        threshold: float = 0.5
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Find potential matches using heuristics.
        Returns list of (bib_key, arxiv_id, score, features)
        """
        matches = []
        
        for bib in bib_entries:
            for arxiv_id, arxiv in arxiv_entries.items():
                features = feature_extractor.extract_features(bib, arxiv)
                
                # Calculate composite score
                score = (
                    features.get('title_jaccard_nostop', 0) * 0.3 +
                    features.get('title_sequence_ratio', 0) * 0.2 +
                    features.get('author_lastname_overlap', 0) * 0.25 +
                    features.get('author_first_match', 0) * 0.1 +
                    features.get('year_exact_match', 0) * 0.1 +
                    features.get('venue_arxiv_id_match', 0) * 0.05
                )
                
                if score >= threshold:
                    matches.append((bib.key, arxiv_id, score, features))
        
        # Sort by score
        matches.sort(key=lambda x: x[2], reverse=True)
        
        return matches


# ================ MAIN ENTRY POINT ================
def main():
    """Main entry point for the reference matching pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reference Matching Pipeline')
    parser.add_argument('--data-dir', type=str, help='Path to data directory containing auto-labeled papers')
    parser.add_argument('--input-dir', type=str, help='Alias for --data-dir (for Colab compatibility)')
    parser.add_argument('--processed-dir', type=str, help='Path to processed directory containing manual labels')
    parser.add_argument('--label-config', type=str, help='Path to label.json configuration file')
    parser.add_argument('--max-auto', type=int, default=None, help='Maximum number of auto publications to load')
    parser.add_argument('--auto-threshold', type=float, default=0.7, help='Threshold for auto-labeling')
    parser.add_argument('--k', type=int, default=5, help='Top-k candidates for ranking')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers (reserved for future use)')
    args = parser.parse_args()
    
    # Resolve directories
    data_dir_str = args.data_dir or args.input_dir
    if not data_dir_str:
        raise SystemExit('Either --data-dir or --input-dir must be provided')
    data_dir = Path(data_dir_str).resolve()
    if not data_dir.exists():
        raise SystemExit(f'Data directory not found: {data_dir}')
    
    processed_dir = Path(args.processed_dir).resolve() if args.processed_dir else None
    if processed_dir is None:
        candidate = data_dir.parent / 'processed'
        if candidate.exists():
            processed_dir = candidate
            logger.info(f"Infer processed directory: {processed_dir}")
    
    if processed_dir and not processed_dir.exists():
        logger.warning(f"Processed directory not found: {processed_dir}")
        processed_dir = None
    
    # Resolve label configuration path
    if args.label_config:
        label_config_path = Path(args.label_config).resolve()
    else:
        label_config_path = data_dir / 'label.json'
    if not label_config_path.exists():
        raise SystemExit(f'label.json not found at {label_config_path}')
    
    logger.info("=" * 50)
    logger.info("Initializing Reference Matching Pipeline")
    logger.info(f"Data directory    : {data_dir}")
    logger.info(f"Processed directory: {processed_dir}")
    logger.info(f"Label config      : {label_config_path}")
    logger.info(f"Auto threshold    : {args.auto_threshold}")
    logger.info(f"Top-k             : {args.k}")
    logger.info(f"Workers           : {args.workers}")
    
    pipeline = ReferenceMatchingPipeline(data_dir, processed_dir)
    pipeline.load_label_config(label_config_path)
    
    logger.info("Loading publications...")
    n_pubs = pipeline.load_all_publications(max_auto=args.max_auto)
    if n_pubs == 0:
        logger.error("No valid publications found!")
        return 1
    
    pipeline.fit_feature_extractor()
    pipeline.create_labels(auto_threshold=args.auto_threshold, load_manual_from_files=True)
    pipeline.auto_split_data()
    pipeline.train_model()
    
    logger.info("Evaluating on validation split...")
    valid_mrr, _ = pipeline.evaluate('valid', k=args.k)
    logger.info("Evaluating on test split...")
    test_mrr, _ = pipeline.evaluate('test', k=args.k)
    
    pipeline.generate_and_save_predictions(k=args.k)
    
    logger.info("=" * 50)
    logger.info(f"Validation MRR@{args.k}: {valid_mrr:.4f}")
    logger.info(f"Test MRR@{args.k}      : {test_mrr:.4f}")
    logger.info("Prediction files written to publication folders")
    logger.info("Pipeline completed successfully")
    
    return 0


if __name__ == "__main__":
    exit(main())
