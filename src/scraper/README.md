# arXiv Scientific Paper ETL Resolver

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Requests](https://img.shields.io/badge/Requests-HTTP-green?style=for-the-badge&logo=python&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-Web%20Scraping-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

> *Faculty of Information Technology, VNU-HCM University of Science*
>
> **Student:** Lê Hà Thanh Chương - 23120195

---

## Table of Contents

- [arXiv Scientific Paper ETL Resolver](#arxiv-scientific-paper-etl-resolver)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
    - [Technical Characteristics](#technical-characteristics)
  - [2. System Architecture](#2-system-architecture)
    - [Data Flow](#data-flow)
  - [3. Data Sources](#3-data-sources)
    - [arXiv ID Format](#arxiv-id-format)
  - [4. Key Features](#4-key-features)
    - [Comprehensive Metadata Collection](#comprehensive-metadata-collection)
    - [Multi-version Processing](#multi-version-processing)
    - [Intelligent Recursive Extraction](#intelligent-recursive-extraction)
    - [Data Cleaning](#data-cleaning)
    - [Reference Collection](#reference-collection)
    - [Parallel Processing](#parallel-processing)
    - [Performance Monitoring](#performance-monitoring)
  - [5. Directory Structure](#5-directory-structure)
  - [6. Installation Guide](#6-installation-guide)
    - [System Requirements](#system-requirements)
    - [Environment Setup](#environment-setup)
    - [Configuration](#configuration)
  - [7. Usage](#7-usage)
    - [Running the Scraper](#running-the-scraper)
    - [Execution Process](#execution-process)
    - [Console Output](#console-output)
  - [8. Output Structure](#8-output-structure)
    - [Per-Paper Output Structure](#per-paper-output-structure)
  - [9. Performance Monitoring](#9-performance-monitoring)
    - [Performance CSV](#performance-csv)
    - [Performance Report](#performance-report)
    - [Performance Benchmark](#performance-benchmark)
  - [Important Notes](#important-notes)
  - [License \& Acknowledgments](#license--acknowledgments)

---

## 1. Project Overview

Dự án này triển khai một *hệ thống thu thập và xử lý dữ liệu tự động (ETL Pipeline)* chuyên biệt cho các bài báo khoa học từ kho lưu trữ *arXiv* – một trong những nguồn tài liệu học thuật mở lớn nhất thế giới. Hệ thống được thiết kế để:

- **Extract:** Thu thập mã nguồn LaTeX, metadata và thông tin trích dẫn từ arXiv và Semantic Scholar API.
- **Transform:** Xử lý các tệp nén đa tầng, lọc tệp LaTeX hợp lệ, loại bỏ các tệp hình ảnh không cần thiết.
- **Load:** Lưu trữ dữ liệu đã xử lý theo cấu trúc thư mục có tổ chức với metadata và references dạng JSON.

### Technical Characteristics

Hệ thống được xây dựng theo *kiến trúc module hóa* với các nguyên tắc thiết kế phần mềm rõ ràng:

```
Data Collection ➔ Multi-source Integration ➔ Parallel Processing ➔ Quality Assurance ➔ Performance Monitoring
```

**Điểm nổi bật:**

- **Xử lý đa luồng (Multi-threading):** Tối ưu hóa tốc độ với ThreadPoolExecutor
- **Tích hợp đa nguồn:** Kết hợp dữ liệu từ arXiv API, arXiv eprint, và Semantic Scholar API
- **Xử lý phiên bản:** Tự động phát hiện và tải tất cả các phiên bản (v1, v2, v3...) của mỗi bài báo
- **Giải nén đệ quy:** Xử lý các tệp nén lồng nhau (tar, tar.gz, gzip) với độ sâu tối đa 3 tầng
- **Rate limiting thông minh:** Đảm bảo tuân thủ giới hạn API với cơ chế thread-safe
- **Theo dõi hiệu năng:** Ghi lại thời gian xử lý, sử dụng bộ nhớ RAM, kích thước dữ liệu

---

## 2. System Architecture

Dự án được tổ chức thành *6 module chức năng* độc lập, đảm bảo tính dễ bảo trì và mở rộng:

| Module | Trách nhiệm | Công nghệ |
| :--- | :--- | :--- |
| **`config.py`** | Quản lý cấu hình tập trung (API keys, paths, parameters) | Python `dotenv`, `pathlib` |
| **`http_client.py`** | Quản lý HTTP session với retry logic và rate limiting thread-safe | `requests`, `urllib3.Retry` |
| **`data_fetchers.py`** | Thu thập dữ liệu từ arXiv (metadata, source) và Semantic Scholar (venue, references) | `requests`, `BeautifulSoup4` |
| **`file_processor.py`** | Giải nén đệ quy, phát hiện tệp LaTeX, lọc và sao chép tệp `.tex` / `.bib` | `tarfile`, `gzip`, `regex` |
| **`workflow.py`** | Điều phối quy trình xử lý song song, tổng hợp kết quả, ghi báo cáo hiệu năng | `ThreadPoolExecutor`, `psutil` |
| **`main.py`** | Entry point khởi động hệ thống | Python standard library |

### Data Flow

```
[arXiv ID Range] 
    ↓
[Batch Processing] → [ThreadPoolExecutor: MAX_WORKERS threads]
    ↓
[Per-Paper Pipeline]
    ├─→ [Metadata Extraction] ← arXiv /abs page
    ├─→ [Version Discovery] ← HTML parsing
    ├─→ [Source Download] ← arXiv /e-print
    ├─→ [Reference Collection] ← Semantic Scholar API
    ├─→ [Archive Extraction] ← Recursive tar/gzip processing
    ├─→ [File Filtering] ← LaTeX detection & image removal
    └─→ [Output Generation] → JSON + Performance CSV
```

---

## 3. Data Sources

Hệ thống tích hợp dữ liệu từ **2 nguồn chính**:

| Nguồn dữ liệu | URL | Thông tin thu thập |
| :--- | :--- | :--- |
| **arXiv** | [https://arxiv.org](https://arxiv.org) | • Metadata (title, authors, dates, category)<br>• LaTeX source files (.tex, .bib)<br>• Version history |
| **Semantic Scholar** | [https://api.semanticscholar.org](https://api.semanticscholar.org) | • Publication venue<br>• Reference metadata (titles, authors, IDs, years) |

### arXiv ID Format

Dự án xử lý các arXiv ID theo định dạng **YYMM.NNNNN** (ví dụ: `2412.15272`):

- **YY:** Năm (2 chữ số)
- **MM:** Tháng (2 chữ số)
- **NNNNN:** Số thứ tự bài báo (5 chữ số)
- **vN:** Phiên bản (tùy chọn, ví dụ: v1, v2, v3...)

---

## 4. Key Features

### Comprehensive Metadata Collection

- Scraping metadata từ trang arXiv `/abs` bằng BeautifulSoup4
- Trích xuất: title, authors, submission date, revision dates, primary category
- Tích hợp publication venue từ Semantic Scholar

### Multi-version Processing

- Tự động phát hiện tất cả các phiên bản của bài báo (v1, v2, v3...)
- Tải và xử lý từng phiên bản riêng biệt
- Lưu trữ theo cấu trúc thư mục phân cấp: `paper_id/tex/version_N/`

### Intelligent Recursive Extraction

- Hỗ trợ nhiều định dạng: `.tar`, `.tar.gz`, `.gz`
- Xử lý các tệp nén lồng nhau với độ sâu tối đa 3 tầng
- Phát hiện tự động tệp LaTeX bằng pattern matching và binary detection

### Data Cleaning

- Loại bỏ tệp hình ảnh (`.png`, `.jpg`, `.pdf`, `.eps`, `.svg`...)
- Chỉ giữ lại tệp nguồn LaTeX (`.tex`) và bibliography (`.bib`)
- Giảm kích thước lưu trữ trung bình **60-80%**

### Reference Collection

- Trích xuất danh sách tài liệu tham khảo từ Semantic Scholar API
- Lọc chỉ các references có arXiv ID
- Lưu metadata chi tiết: title, authors, year, Semantic Scholar ID

### Parallel Processing

- Sử dụng ThreadPoolExecutor với số lượng worker có thể cấu hình
- Xử lý theo batch để tối ưu hiệu suất
- Thread-safe rate limiting để tuân thủ API quotas

### Performance Monitoring

- Ghi lại thời gian xử lý từng bài báo
- Theo dõi sử dụng RAM (max, average) bằng `psutil`
- Đo lường kích thước dữ liệu trước/sau xử lý
- Tạo báo cáo tổng hợp: success rate, throughput, resource usage

---

## 5. Directory Structure

```
scientific-paper-etl-resolver/
│
├── src/
│   └── scraper/
│       ├── config.py              # Cấu hình tập trung
│       ├── http_client.py         # HTTP session & rate limiting
│       ├── data_fetchers.py       # API calls & web scraping
│       ├── file_processor.py      # Archive extraction & filtering
│       ├── workflow.py            # Orchestration & parallelization
│       ├── main.py                # Entry point
│       ├── utils.py               # Helper functions
│       ├── requirements.txt       # Dependencies
│       └── README1.md             # Documentation
│
├── {STUDENT_ID}/                  # Output directory (e.g., "23120195/")
│   ├── {paper_id_1}/
│   │   ├── metadata.json          # Paper metadata
│   │   ├── references.json        # Reference list
│   │   └── tex/
│   │       ├── v1/                # Version 1 LaTeX files
│   │       │   ├── main.tex
│   │       │   └── references.bib
│   │       └── v2/                # Version 2 LaTeX files
│   │           └── ...
│   ├── {paper_id_2}/
│   │   └── ...
│   ├── performance.csv            # Per-paper metrics
│   └── performance_report.json    # Summary statistics
│
├── data/                          # Data directories
│   ├── raw/                       # Raw data
│   └── processed/                 # Processed data
│
├── notebooks/                     # Jupyter notebooks
├── reports/                       # Analysis reports
├── requirements.txt               # Project dependencies
└── README.md                      # Project overview
```

---

## 6. Installation Guide

### System Requirements

- **Python:** 3.8 hoặc cao hơn
- **Hệ điều hành:** Windows / Linux / macOS
- **RAM:** Tối thiểu 4GB (khuyến nghị 8GB+ cho xử lý đa luồng)
- **Dung lượng ổ cứng:** Phụ thuộc vào số lượng bài báo (khoảng 5-20MB/paper)

### Environment Setup

**Bước 1: Clone repository**
```bash
git clone <repository-url>
cd scientific-paper-etl-resolver
```

**Bước 2: Tạo môi trường ảo**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

**Bước 3: Cài đặt thư viện**
```bash
cd src/scraper
pip install -r requirements.txt
```

**Dependencies chính:**
- `requests` - HTTP client
- `beautifulsoup4` - HTML parsing
- `lxml` - XML/HTML parser
- `psutil` - Process monitoring
- `python-dotenv` - Environment variable management

### Configuration

**Bước 4: Tạo file `.env`**

Tạo file `.env` trong thư mục `src/scraper/` với nội dung:

```bash
SEMANTIC_SCHOLAR_API_KEY=your_api_key_here
STUDENT_ID=23120195
```

> **Lưu ý:** Đăng ký API key miễn phí tại [Semantic Scholar API](https://www.semanticscholar.org/product/api)

**Bước 5: Tùy chỉnh tham số trong `config.py`**

| Tham số | Kiểu dữ liệu | Mô tả | Giá trị mặc định |
| :--- | :---: | :--- | :---: |
| `STUDENT_ID` | `str` | Mã số sinh viên (tên thư mục output) | `"23120195"` |
| `START_MONTH` | `str` | Tháng bắt đầu (định dạng YYMM) | `"2412"` |
| `START_ID` | `int` | arXiv ID bắt đầu | `15272` |
| `END_ID` | `int` | arXiv ID kết thúc | `15274` |
| `MAX_WORKERS` | `int` | Số luồng xử lý song song | `5` |
| `S2_DELAY` | `float` | Độ trễ giữa các API call (giây) | `0.5` |
| `BATCH_SIZE` | `int` | Số bài báo mỗi batch | `50` |

**Khuyến nghị:**
- `MAX_WORKERS`: 3-5 cho máy cá nhân, 10-20 cho server
- `S2_DELAY`: ≥ 0.5s để tránh bị rate limit (với API key: ≥ 0.1s)
- `BATCH_SIZE`: 20-50 papers tùy theo RAM khả dụng

---

## 7. Usage

### Running the Scraper

```bash
# Từ thư mục gốc của project
python src/scraper/main.py
```

### Execution Process

Hệ thống sẽ tự động thực hiện các bước sau:

1. **Khởi tạo:** Tạo thư mục output, kiểm tra cấu hình
2. **Batch Processing:** Chia danh sách arXiv IDs thành các batch
3. **Parallel Execution:** Xử lý mỗi batch với nhiều worker threads
4. **Per-Paper Pipeline:**
   - Lấy metadata từ arXiv
   - Phát hiện các phiên bản
   - Tải mã nguồn LaTeX
   - Lấy thông tin references từ Semantic Scholar
   - Giải nén và lọc tệp
   - Lưu kết quả và ghi log hiệu năng
5. **Finalization:** Tạo báo cáo tổng hợp

### Console Output

```
[14:23:45] --- STARTING SCRAPING OF 1000 PAPERS ---
[14:23:45] Configuration: 5 workers, batch size 50
[14:23:46] Start processing 2412.15272
[14:23:47] [OK] Retrieved venue for 2412.15272
[14:23:47] [OK] Retrieved 23 references for 2412.15272
[14:23:48] Discovered versions for 2412.15272: ['v1', 'v2']
[14:23:49] [OK] Downloaded 2412-15272v1 (1234.5 KB) type=application/x-gzip
[14:23:50] [EXTRACTION RESULT] Total files: 45, .tex: 3, .bib: 1
[14:23:50] [SUMMARY] Total copied: 3 .tex files, 1 .bib files
...
[15:10:23] --- SCRAPING COMPLETED ---
[15:10:23] Successful papers: 987/1000 (98.7%)
[15:10:23] Total time: 46.6 minutes
```

---

## 8. Output Structure

### Per-Paper Output Structure

**`{STUDENT_ID}/{paper_id}/metadata.json`**
```json
{
  "paper_title": "Attention Is All You Need",
  "authors": ["Ashish Vaswani", "Noam Shazeer", ...],
  "submission_date": "2017-06-12",
  "revised_dates": ["2017-06-12", "2017-08-02"],
  "publication_venue": "NeurIPS"
}
```

**`{STUDENT_ID}/{paper_id}/references.json`**
```json
{
  "1706-03762": {
    "paper_title": "Neural Machine Translation by Jointly Learning to Align and Translate",
    "authors": ["Dzmitry Bahdanau", "Kyunghyun Cho", "Yoshua Bengio"],
    "submission_date": "2014-09-01",
    "semantic_scholar_id": "a7a2734e2d2f17e3e0a9b1c4d5e6f7g8"
  },
  ...
}
```

**`{STUDENT_ID}/{paper_id}/tex/v1/`**
- Chứa tất cả tệp `.tex` và `.bib` từ phiên bản 1
- Cấu trúc thư mục giữ nguyên như bản gốc

---

## 9. Performance Monitoring

### Performance CSV

File `performance.csv` ghi lại chi tiết từng bài báo:

| Cột | Mô tả |
| :--- | :--- |
| `arxiv_id` | ID của bài báo |
| `versions_found` | Danh sách các phiên bản |
| `tex_files` | Số tệp `.tex` tìm được |
| `bib_files` | Số tệp `.bib` tìm được |
| `references_count` | Số references từ Semantic Scholar |
| `size_before_bytes` | Kích thước trước khi xóa hình ảnh |
| `size_after_bytes` | Kích thước sau khi xóa hình ảnh |
| `reduction_percent` | Phần trăm giảm dung lượng |
| `duration_seconds` | Thời gian xử lý |
| `max_ram_mb` | RAM tối đa sử dụng |
| `avg_ram_mb` | RAM trung bình |
| `status` | `success` hoặc `no_tex` |

### Performance Report

File `performance_report.json` chứa thống kê tổng hợp:

```json
{
  "performance_metrics": {
    "total_papers": 1000,
    "successful_papers": 987,
    "failed_papers": 13,
    "success_rate": "98.7%",
    "total_processing_time_seconds": 2796.3,
    "total_processing_time_minutes": 46.6,
    "papers_per_second": 0.358,
    "papers_per_minute": 21.5,
    "total_memory_usage_mb": 4523.7,
    "peak_memory_usage_mb": 856.2,
    "avg_references_per_paper": 18.4,
    "reference_metadata_success_rate": "94.3%",
    "configuration": {
      "max_workers": 5,
      "batch_size": 50,
      "s2_delay": 0.5
    }
  }
}
```

### Performance Benchmark

**Cấu hình thử nghiệm:**
- CPU: Intel i7-10700K (8 cores, 16 threads)
- RAM: 16GB DDR4
- Network: 100 Mbps
- Parameters: MAX_WORKERS=5, BATCH_SIZE=50

**Kết quả:** 
- **Throughput:** ~21 papers/minute
- **Success Rate:** 98.7%
- **Avg. RAM Usage:** ~450 MB
- **Storage Reduction:** 65% (trung bình sau khi xóa images)

---

## Important Notes

1. **Rate Limiting:** Luôn tuân thủ `S2_DELAY` ≥ 0.5s khi không có API key, ≥ 0.1s khi có key
2. **API Key:** Đăng ký miễn phí tại [Semantic Scholar](https://www.semanticscholar.org/product/api) để tăng quota
3. **Error Handling:** Hệ thống tự động retry khi gặp lỗi HTTP 429/500/502/503/504
4. **Storage:** Dự trù khoảng 10-15 MB cho mỗi bài báo (sau khi xóa images)
5. **Network:** Yêu cầu kết nối Internet ổn định trong suốt quá trình chạy

---

## License & Acknowledgments

- **arXiv:** Dữ liệu được sử dụng tuân theo [arXiv Terms of Use](https://arxiv.org/help/api/tou)
- **Semantic Scholar:** API được cấp phép cho mục đích nghiên cứu học thuật
- **Course:** Introduction to Data Science - VNU-HCM University of Science

---

**Developed by:** Lê Hà Thanh Chương - 23120195  
**Last Updated:** February 2026