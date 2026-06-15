"""
URL ingestion module for web content extraction and processing.
Provides functions to fetch, parse, clean, and chunk web content for RAG systems.
"""

import re
import time
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def validate_url(url: str) -> bool:
    """Validate URL format.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    # Check if URL starts with http:// or https://
    if not url.startswith(('http://', 'https://')):
        return False
    
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def fetch_url_content(url: str, timeout: int = 5) -> Tuple[str, Dict[str, str]]:
    """Fetch raw HTML content from URL.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (html_content, metadata_dict)
        
    Raises:
        ValueError: If URL is invalid
        requests.RequestException: If fetch fails
    """
    if not validate_url(url):
        raise ValueError(f"Invalid URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Extract metadata
        metadata = {
            'url': url,
            'status_code': response.status_code,
            'fetch_time': datetime.now().isoformat(),
            'domain': urlparse(url).netloc
        }
        
        return response.text, metadata
        
    except requests.Timeout:
        raise requests.RequestException(f"Timeout fetching {url} after {timeout} seconds")
    except requests.HTTPError as e:
        raise requests.RequestException(f"HTTP error fetching {url}: {e}")
    except requests.RequestException as e:
        raise requests.RequestException(f"Error fetching {url}: {e}")


def extract_text_from_html(html: str, url: str) -> Tuple[str, Dict[str, str]]:
    """Extract clean text from HTML content.
    
    Args:
        html: Raw HTML content
        url: Source URL for metadata
        
    Returns:
        Tuple of (cleaned_text, metadata_dict)
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove unwanted elements
    for element in soup(['script', 'style', 'noscript', 'iframe', 'nav', 'footer', 'header', 'aside']):
        element.decompose()
    
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, type(soup.find(string=True))) and text.strip().startswith('<!--')):
        comment.extract()
    
    # Extract title
    title = soup.title.string if soup.title else ''
    title = title.strip() if title else urlparse(url).netloc
    
    # Extract meta description
    meta_desc = ''
    meta_tag = soup.find('meta', attrs={'name': 'description'})
    if meta_tag and meta_tag.get('content'):
        meta_desc = meta_tag['content'].strip()
    
    # Get main content - try to find article or main tag
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|article|post', re.I))
    
    if main_content:
        text = main_content.get_text(separator='\n', strip=True)
    else:
        # Fallback to body
        text = soup.body.get_text(separator='\n', strip=True) if soup.body else ''
    
    # Clean up text
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]  # Remove empty lines
    cleaned_text = '\n'.join(lines)
    
    # Build metadata
    metadata = {
        'source_type': 'url',
        'url': url,
        'domain': urlparse(url).netloc,
        'original_title': title,
        'description': meta_desc,
        'fetch_time': datetime.now().isoformat()
    }
    
    return cleaned_text, metadata


def chunk_text(text: str, metadata: Dict[str, str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Chunk text into smaller pieces with metadata preservation.
    
    Args:
        text: Text to chunk
        metadata: Metadata to attach to each chunk
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects with chunks
    """
    if len(text.strip()) < 100:
        raise ValueError(f"Text too short after cleaning ({len(text.strip())} chars, minimum 100)")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=['\n\n', '\n', '. ', ' ', '']
    )
    
    chunks = text_splitter.split_text(text)
    
    # Create Document objects with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = metadata.copy()
        chunk_metadata['chunk_index'] = i
        chunk_metadata['total_chunks'] = len(chunks)
        documents.append(Document(page_content=chunk, metadata=chunk_metadata))
    
    return documents


def process_url(url: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[List[Document], Dict[str, str]]:
    """Process a single URL through the full pipeline.
    
    Args:
        url: URL to process
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        Tuple of (list of Document chunks, processing_metadata)
        
    Raises:
        ValueError: If URL is invalid or content too short
        requests.RequestException: If fetch fails
    """
    # Fetch HTML
    html, fetch_metadata = fetch_url_content(url)
    
    # Extract text
    cleaned_text, text_metadata = extract_text_from_html(html, url)
    
    # Chunk text
    documents = chunk_text(cleaned_text, text_metadata, chunk_size, chunk_overlap)
    
    # Combine metadata
    processing_metadata = {
        **fetch_metadata,
        **text_metadata,
        'chunks_created': len(documents),
        'original_length': len(cleaned_text)
    }
    
    return documents, processing_metadata


def ingest_urls(urls_list: List[str], existing_urls: Optional[set] = None, 
                chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, any]:
    """Batch process multiple URLs.
    
    Args:
        urls_list: List of URLs to process
        existing_urls: Set of already processed URLs to skip duplicates
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        Dictionary with processing results:
        {
            'success': List of successfully processed URLs,
            'failed': List of failed URLs with error messages,
            'skipped': List of skipped (duplicate) URLs,
            'total_chunks': Total chunks created,
            'all_documents': List of all Document objects
        }
    """
    if existing_urls is None:
        existing_urls = set()
    
    results = {
        'success': [],
        'failed': [],
        'skipped': [],
        'total_chunks': 0,
        'all_documents': []
    }
    
    for url in urls_list:
        try:
            # Skip duplicates
            if url in existing_urls:
                results['skipped'].append(url)
                continue
            
            # Process URL
            documents, metadata = process_url(url, chunk_size, chunk_overlap)
            
            results['success'].append({
                'url': url,
                'chunks': len(documents),
                'metadata': metadata
            })
            
            results['all_documents'].extend(documents)
            results['total_chunks'] += len(documents)
            existing_urls.add(url)
            
            # Small delay to be respectful
            time.sleep(0.5)
            
        except ValueError as e:
            results['failed'].append({'url': url, 'error': str(e)})
        except requests.RequestException as e:
            results['failed'].append({'url': url, 'error': str(e)})
        except Exception as e:
            results['failed'].append({'url': url, 'error': f"Unexpected error: {str(e)}"})
    
    return results
