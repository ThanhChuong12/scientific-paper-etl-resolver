"""
Reference Matching Pipeline
============================
This module implements a complete reference matching pipeline that matches
BibTeX entries (from refs.bib) with arXiv entries (from references.json).

Problem Type: RANKING/RETRIEVAL
- Input: A BibTeX entry (from refs.bib)
- Output: Ranked list of top 5 arXiv IDs from references.json
- Metric: Mean Reciprocal Rank (MRR@5)
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