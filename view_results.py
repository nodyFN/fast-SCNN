#!/usr/bin/env python3
"""
Fast-SCNN Inference Results Web Viewer.
A zero-dependency local web server to browse and debug segmentation outputs.

Usage
-----
# Default directory (inference_results/)
python view_results.py

# Custom directory
python view_results.py --dir results/
"""

import argparse
import http.server
import json
import logging
import socketserver
import sys
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PORT = 8000

# Mapping of file suffixes to descriptive display names
SUFFIX_MAP = {
    "_overlay.jpg": "Prediction Overlay",
    "_comparison.jpg": "Error Diagnostic (TP/FP/FN)",
    "_prob.jpg": "Saliency Heatmap",
    "_binary.png": "Binary Mask (0-255)",
    "_class.png": "Class Mask (0-1)",
    "_prob_gray.png": "Grayscale Probability",
    "_merged.jpg": "Merged Collage"
}


def scan_inference_directory(dir_path: Path) -> List[Dict]:
    """Scan directory and group generated files by image stem."""
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {dir_path}")
        return []

    # Find all files
    files = sorted(list(dir_path.iterdir()))
    
    # Group by stem
    groups: Dict[str, Dict[str, str]] = {}
    
    for f in files:
        if f.is_dir() or f.name.startswith("."):
            continue
        
        # Check matching suffixes
        matched = False
        for suffix, display_name in SUFFIX_MAP.items():
            if f.name.endswith(suffix):
                stem = f.name[:-len(suffix)]
                if stem not in groups:
                    groups[stem] = {}
                groups[stem][suffix] = f.name
                matched = True
                break
        
        # Fallback for unrecognized image files (e.g. input images copied manually)
        if not matched and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            stem = f.stem
            if stem not in groups:
                groups[stem] = {}
            # Use raw suffix
            groups[stem][f.suffix] = f.name

    # Convert to sorted list of groups
    sorted_stems = sorted(groups.keys())
    result = []
    for stem in sorted_stems:
        # Re-map keys for frontend convenience
        file_list = {}
        for suffix, filename in groups[stem].items():
            key = suffix.lstrip("_").replace(".", "_")
            file_list[key] = filename
        result.append({
            "stem": stem,
            "files": file_list
        })
    return result


def create_handler(results_dir: Path):
    class ResultsViewerHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress default noisy http request logging
            pass

        def do_GET(self):
            parsed_url = urllib.parse.urlparse(self.path)
            path_str = parsed_url.path

            # API: get list of groups
            if path_str == "/api/groups":
                groups = scan_inference_directory(results_dir)
                response_data = json.dumps(groups).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_data)))
                self.end_headers()
                self.wfile.write(response_data)
                return

            # Main SPA Dashboard
            if path_str in {"/", "/index.html"}:
                html_content = self.get_index_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(html_content)))
                self.end_headers()
                self.wfile.write(html_content)
                return

            # Serve files from the inference results directory
            requested_file = results_dir / urllib.parse.unquote(path_str.lstrip("/"))
            # Prevent directory traversal attacks
            try:
                resolved_requested = requested_file.resolve()
                resolved_base = results_dir.resolve()
                is_subdir = resolved_requested.relative_to(resolved_base)
            except Exception:
                is_subdir = False

            if requested_file.exists() and requested_file.is_file() and is_subdir:
                self.send_response(200)
                if requested_file.suffix.lower() in {".jpg", ".jpeg"}:
                    self.send_header("Content-Type", "image/jpeg")
                elif requested_file.suffix.lower() == ".png":
                    self.send_header("Content-Type", "image/png")
                else:
                    self.send_header("Content-Type", "application/octet-stream")
                
                with open(requested_file, "rb") as f:
                    content = f.read()
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                return

            # Fallback 404
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File not found")

        def get_index_html(self) -> str:
            """Sleek SPA dashboard frontend for viewing diagnostic maps."""
            return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fast-SCNN Inference Diagnostics Viewer</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --border-color: #334155;
            --text-color: #f1f5f9;
            --text-secondary: #94a3b8;
            --primary: #10b981;
            --primary-hover: #059669;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background-color: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border-color);
            padding: 1.25rem 2rem;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .header-content {
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .logo h1 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--primary);
            letter-spacing: -0.025em;
        }
        
        .logo p {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .controls {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        button {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.875rem;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s ease;
        }

        button:hover:not(:disabled) {
            border-color: var(--primary);
            background-color: rgba(16, 185, 129, 0.1);
        }

        button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        select {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            outline: none;
            font-family: inherit;
            font-size: 0.875rem;
            cursor: pointer;
            min-width: 250px;
            transition: border-color 0.2s ease;
        }

        select:focus {
            border-color: var(--primary);
        }

        .config-panel {
            background-color: rgba(30, 41, 59, 0.4);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
        }

        .config-container {
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 1.5rem;
        }

        .panel-label {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .toggles {
            display: flex;
            align-items: center;
            gap: 1.25rem;
            flex-wrap: wrap;
        }

        .toggle-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
            font-size: 0.875rem;
            user-select: none;
        }

        .toggle-item input[type="checkbox"] {
            accent-color: var(--primary);
            width: 1rem;
            height: 1rem;
            cursor: pointer;
        }

        main {
            flex: 1;
            padding: 2rem;
            max-width: 1800px;
            width: 100%;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .group-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }

        .group-title {
            font-size: 1.5rem;
            font-weight: 600;
        }

        .group-meta {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 1.5rem;
            transition: all 0.3s ease;
        }

        .image-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .image-card:hover {
            border-color: rgba(16, 185, 129, 0.4);
            transform: translateY(-2px);
        }

        .card-header {
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-header .badge {
            background-color: var(--border-color);
            font-size: 0.75rem;
            padding: 0.15rem 0.5rem;
            border-radius: 10px;
            color: var(--text-secondary);
        }

        .image-container {
            position: relative;
            width: 100%;
            padding-top: 56.25%; /* 16:9 Aspect Ratio */
            background-color: #0b0f19;
            overflow: hidden;
        }

        .image-container img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            cursor: zoom-in;
            transition: transform 0.3s ease;
        }

        /* Fullscreen Modal View */
        .modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: rgba(15, 23, 42, 0.95);
            z-index: 1000;
            display: none;
            justify-content: center;
            align-items: center;
            cursor: zoom-out;
        }

        .modal img {
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .empty-state {
            text-align: center;
            padding: 5rem 2rem;
            color: var(--text-secondary);
            border: 2px dashed var(--border-color);
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }

        .empty-state h3 {
            color: var(--text-color);
            font-size: 1.25rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <div class="logo">
                <h1>FAST-SCNN DIAGNOSTICS</h1>
                <p>Interactive Output Debugging Dashboard</p>
            </div>
            <div class="controls">
                <button id="prevBtn" onclick="navigate(-1)">&larr; Prev</button>
                <select id="groupSelect" onchange="selectGroup(this.value)">
                    <option value="">Loading image groups...</option>
                </select>
                <button id="nextBtn" onclick="navigate(1)">Next &rarr;</button>
            </div>
        </div>
    </header>

    <div class="config-panel">
        <div class="config-container">
            <span class="panel-label">Active Layers:</span>
            <div class="toggles" id="togglesContainer">
                <!-- Checkboxes will be populated dynamically -->
            </div>
        </div>
    </div>

    <main>
        <div id="contentArea">
            <div class="group-header">
                <span class="group-title" id="currentGroupTitle">Select an image</span>
                <span class="group-meta" id="currentGroupMeta">Group 0 of 0</span>
            </div>
            <div class="image-grid" id="imageGrid" style="margin-top: 1.5rem;">
                <!-- Image cards will be rendered dynamically -->
            </div>
        </div>
        <div class="empty-state" id="emptyState" style="display: none;">
            <h3>No Diagnostic Images Found</h3>
            <p>Make sure you run inference first. The tool scans files inside the output directory.</p>
        </div>
    </main>

    <div class="modal" id="imageModal" onclick="closeModal()">
        <img id="modalImg" src="" alt="Fullscreen Map">
    </div>

    <script>
        let groups = [];
        let currentIndex = -1;
        
        // Dictionary of maps with corresponding suffix key and display titles
        const MAP_DEFS = {
            "overlay_jpg": { name: "Prediction Overlay", default: true },
            "comparison_jpg": { name: "Error Diagnostic (TP/FP/FN)", default: true },
            "prob_jpg": { name: "Saliency Heatmap", default: true },
            "binary_png": { name: "Binary Mask (0-255)", default: false },
            "class_png": { name: "Class Mask (0-1)", default: false },
            "prob_gray_png": { name: "Grayscale Probability", default: false },
            "merged_jpg": { name: "Merged Collage", default: false }
        };

        // Initialize display configuration in local storage
        let displayConfigs = JSON.parse(localStorage.getItem('viewer_toggles')) || {};
        if (Object.keys(displayConfigs).length === 0) {
            // Apply defaults
            Object.keys(MAP_DEFS).forEach(k => {
                displayConfigs[k] = MAP_DEFS[k].default;
            });
        }

        // Fetch groups from local server API
        async function fetchGroups() {
            try {
                const response = await fetch('/api/groups');
                groups = await response.json();
                
                if (groups.length === 0) {
                    showEmptyState();
                    return;
                }
                
                hideEmptyState();
                populateDropdown();
                renderToggles();
                
                // Select first group by default
                selectIndex(0);
            } catch (err) {
                console.error("Failed to load image groups", err);
                showEmptyState();
            }
        }

        function showEmptyState() {
            document.getElementById('contentArea').style.display = 'none';
            document.getElementById('emptyState').style.display = 'flex';
            document.getElementById('prevBtn').disabled = true;
            document.getElementById('nextBtn').disabled = true;
        }

        function hideEmptyState() {
            document.getElementById('contentArea').style.display = 'block';
            document.getElementById('emptyState').style.display = 'none';
        }

        function populateDropdown() {
            const select = document.getElementById('groupSelect');
            select.innerHTML = '';
            groups.forEach((g, idx) => {
                const opt = document.createElement('option');
                opt.value = idx;
                opt.textContent = `${idx + 1}. ${g.stem}`;
                select.appendChild(opt);
            });
        }

        function renderToggles() {
            const container = document.getElementById('togglesContainer');
            container.innerHTML = '';
            
            Object.keys(MAP_DEFS).forEach(k => {
                const label = document.createElement('label');
                label.className = 'toggle-item';
                
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.checked = displayConfigs[k];
                cb.onchange = (e) => {
                    displayConfigs[k] = e.target.checked;
                    localStorage.setItem('viewer_toggles', JSON.stringify(displayConfigs));
                    renderImages();
                };
                
                label.appendChild(cb);
                label.appendChild(document.createTextNode(MAP_DEFS[k].name));
                container.appendChild(label);
            });
        }

        function selectIndex(idx) {
            if (idx < 0 || idx >= groups.length) return;
            currentIndex = idx;
            
            // Update controls
            document.getElementById('groupSelect').value = idx;
            document.getElementById('prevBtn').disabled = (idx === 0);
            document.getElementById('nextBtn').disabled = (idx === groups.length - 1);
            
            // Meta details
            const group = groups[currentIndex];
            document.getElementById('currentGroupTitle').textContent = group.stem;
            document.getElementById('currentGroupMeta').textContent = `Image ${idx + 1} of ${groups.length}`;
            
            renderImages();
        }

        function selectGroup(val) {
            if (val !== "") {
                selectIndex(parseInt(val));
            }
        }

        function navigate(dir) {
            const target = currentIndex + dir;
            if (target >= 0 && target < groups.length) {
                selectIndex(target);
            }
        }

        function renderImages() {
            const grid = document.getElementById('imageGrid');
            grid.innerHTML = '';
            
            if (currentIndex === -1 || groups.length === 0) return;
            
            const group = groups[currentIndex];
            let renderedCount = 0;
            
            Object.keys(MAP_DEFS).forEach(k => {
                // If this toggled map type exists in active group files
                if (displayConfigs[k] && group.files[k]) {
                    const filename = group.files[k];
                    const displayName = MAP_DEFS[k].name;
                    
                    const card = document.createElement('div');
                    card.className = 'image-card';
                    
                    card.innerHTML = `
                        <div class="card-header">
                            <span>${displayName}</span>
                            <span class="badge">${filename.substring(filename.lastIndexOf('.'))}</span>
                        </div>
                        <div class="image-container">
                            <img src="/${filename}" alt="${displayName}" onclick="openModal(this.src)">
                        </div>
                    `;
                    grid.appendChild(card);
                    renderedCount++;
                }
            });

            // Adjust grid columns count dynamically
            if (renderedCount === 1) {
                grid.style.gridTemplateColumns = '1fr';
            } else if (renderedCount === 2) {
                grid.style.gridTemplateColumns = 'repeat(2, 1fr)';
            } else {
                grid.style.gridTemplateColumns = 'repeat(auto-fit, minmax(480px, 1fr))';
            }

            if (renderedCount === 0) {
                grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 3rem; color: var(--text-secondary);">No layers selected. Check some active layers above.</div>';
            }
        }

        // Fullscreen Modal utilities
        function openModal(src) {
            const modal = document.getElementById('imageModal');
            const img = document.getElementById('modalImg');
            img.src = src;
            modal.style.display = 'flex';
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        // Keyboard arrows navigation listener
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                navigate(-1);
            } else if (e.key === 'ArrowRight') {
                navigate(1);
            } else if (e.key === 'Escape') {
                closeModal();
            }
        });

        // Fetch dataset list on load
        fetchGroups();
    </script>
</body>
</html>
"""
    return ResultsViewerHandler


def main() -> None:
    p = argparse.ArgumentParser(description="Start Web Results Viewer")
    p.add_argument("--dir", type=str, default="inference_results",
                   help="Directory containing inference output files")
    p.add_argument("--port", type=int, default=PORT, help="Port to serve viewer on")
    args = p.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.exists():
        logger.error(f"Inference directory '{results_dir}' does not exist! Please run inference.py first.")
        sys.exit(1)

    logger.info(f"Scanning inference results in '{results_dir.resolve()}'")
    groups = scan_inference_directory(results_dir)
    logger.info(f"Discovered {len(groups)} image groups.")

    handler = create_handler(results_dir)
    
    # Allow address reuse to prevent "Address already in use" errors during rapid restarts
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", args.port), handler) as httpd:
            url = f"http://localhost:{args.port}"
            logger.info(f"Starting server at {url}")
            logger.info("Press Ctrl+C to stop.")
            
            # Auto-open browser
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.warning(f"Could not open browser automatically: {e}")
                
            httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user.")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")


if __name__ == "__main__":
    main()
