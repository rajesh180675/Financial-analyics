import streamlit as st
import requests
import io
import zipfile
from urllib.parse import urlparse
import re

def get_raw_github_url(github_url):
    """Convert GitHub URL to raw content URL"""
    # Handle different GitHub URL formats
    if "raw.githubusercontent.com" in github_url:
        return github_url
    
    # Convert regular GitHub URL to raw URL
    # Example: https://github.com/user/repo/blob/main/file.py
    # Becomes: https://raw.githubusercontent.com/user/repo/main/file.py
    pattern = r'https://github\.com/([^/]+)/([^/]+)/blob/(.+)'
    match = re.match(pattern, github_url)
    
    if match:
        user, repo, path = match.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{path}"
    
    return github_url

def download_github_file(url):
    """Download content from GitHub"""
    try:
        raw_url = get_raw_github_url(url)
        response = requests.get(raw_url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading file: {str(e)}")
        return None

def split_file_into_chunks(content, chunk_size=2500):
    """Split file content into chunks of specified line count"""
    lines = content.split('\n')
    total_lines = len(lines)
    chunks = []
    
    for i in range(0, total_lines, chunk_size):
        chunk_lines = lines[i:i + chunk_size]
        chunks.append('\n'.join(chunk_lines))
    
    return chunks, total_lines

def get_filename_from_url(url):
    """Extract filename from URL"""
    path = urlparse(url).path
    filename = path.split('/')[-1]
    if not filename:
        filename = "source_file.py"
    return filename

def create_zip_file(chunks, base_filename):
    """Create a ZIP file containing all chunks"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, chunk in enumerate(chunks, 1):
            filename = f"{base_filename.rsplit('.', 1)[0]}_part_{i}.py"
            zip_file.writestr(filename, chunk)
    
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit UI
st.set_page_config(page_title="GitHub File Splitter", page_icon="✂️", layout="wide")

st.title("🔧 GitHub Python File Splitter")
st.markdown("Split large Python files from GitHub into smaller chunks of 2500 lines each")

# Input section
st.markdown("### 📥 Input")
github_url = st.text_input(
    "Enter GitHub file URL:",
    placeholder="https://github.com/username/repo/blob/main/file.py",
    help="Paste the GitHub URL of the Python file you want to split"
)

# Configuration
col1, col2 = st.columns(2)
with col1:
    chunk_size = st.number_input(
        "Lines per chunk:",
        min_value=100,
        max_value=10000,
        value=2500,
        step=100,
        help="Maximum number of lines in each split file"
    )

# Process button
if st.button("🚀 Process File", type="primary"):
    if github_url:
        with st.spinner("Downloading and processing file..."):
            # Download file
            content = download_github_file(github_url)
            
            if content:
                # Split into chunks
                chunks, total_lines = split_file_into_chunks(content, int(chunk_size))
                
                # Get original filename
                original_filename = get_filename_from_url(github_url)
                base_filename = original_filename.rsplit('.', 1)[0]
                
                # Display results
                st.success(f"✅ File processed successfully!")
                
                # Statistics
                st.markdown("### 📊 Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Lines", f"{total_lines:,}")
                with col2:
                    st.metric("Number of Chunks", len(chunks))
                with col3:
                    st.metric("Lines per Chunk", chunk_size)
                
                # Download section
                st.markdown("### 💾 Download Options")
                
                # Option 1: Download all as ZIP
                zip_buffer = create_zip_file(chunks, base_filename)
                st.download_button(
                    label="📦 Download All Files (ZIP)",
                    data=zip_buffer,
                    file_name=f"{base_filename}_split.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                
                st.markdown("---")
                
                # Option 2: Download individual files
                st.markdown("#### Download Individual Files:")
                
                # Create columns for download buttons (3 per row)
                for i in range(0, len(chunks), 3):
                    cols = st.columns(3)
                    for j in range(3):
                        if i + j < len(chunks):
                            chunk_index = i + j
                            chunk = chunks[chunk_index]
                            lines_in_chunk = len(chunk.split('\n'))
                            filename = f"{base_filename}_part_{chunk_index + 1}.py"
                            
                            with cols[j]:
                                st.download_button(
                                    label=f"📄 Part {chunk_index + 1} ({lines_in_chunk} lines)",
                                    data=chunk,
                                    file_name=filename,
                                    mime="text/plain",
                                    key=f"download_{chunk_index}"
                                )
                
                # Preview section
                st.markdown("### 👁️ Preview")
                preview_option = st.selectbox(
                    "Select chunk to preview:",
                    options=[f"Part {i+1}" for i in range(len(chunks))],
                    index=0
                )
                
                if preview_option:
                    chunk_index = int(preview_option.split()[-1]) - 1
                    st.code(chunks[chunk_index][:1000] + "\n...\n[Preview truncated to first 1000 characters]", 
                           language="python")
    else:
        st.warning("⚠️ Please enter a GitHub URL")

# Instructions
with st.expander("📖 How to use"):
    st.markdown("""
    1. **Find your file on GitHub**: Navigate to the Python file you want to split on GitHub
    2. **Copy the URL**: Copy the URL from your browser's address bar
    3. **Paste the URL**: Paste it in the input field above
    4. **Adjust chunk size**: Optionally adjust the lines per chunk (default is 2500)
    5. **Process**: Click the "Process File" button
    6. **Download**: Choose to download all files as a ZIP or download individual chunks
    
    **Supported URL formats:**
    - `https://github.com/username/repo/blob/main/file.py`
    - `https://raw.githubusercontent.com/username/repo/main/file.py`
    
    **Note:** The file must be publicly accessible on GitHub.
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Split large Python files into manageable chunks")
