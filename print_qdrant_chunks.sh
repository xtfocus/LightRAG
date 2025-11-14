#!/bin/bash

# Script to print Qdrant raw chunks for a given document ID
# Usage: ./print_qdrant_chunks.sh <document_id> [workspace] [qdrant_url] [api_key]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
RAW_OUTPUT=false
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [OPTIONS] <document_id> [workspace] [qdrant_url] [api_key]"
    echo ""
    echo "Print Qdrant raw chunks for a given document ID"
    echo ""
    echo "Arguments:"
    echo "  document_id  - The document ID to query chunks for (required)"
    echo "  workspace    - Workspace identifier (default: '_' or from QDRANT_WORKSPACE env var)"
    echo "  qdrant_url   - Qdrant server URL (default: http://localhost:6333 or from QDRANT_URL env var)"
    echo "  api_key      - Qdrant API key (optional, from QDRANT_API_KEY env var if not provided)"
    echo ""
    echo "Options:"
    echo "  -h, --help   - Show this help message"
    echo "  -r, --raw    - Output only raw chunk payloads (one JSON object per line)"
    echo ""
    echo "Environment variables:"
    echo "  QDRANT_URL       - Qdrant server URL"
    echo "  QDRANT_API_KEY   - Qdrant API key"
    echo "  QDRANT_WORKSPACE - Workspace identifier"
    exit 0
fi

# Check for raw output flag
if [ "$1" = "--raw" ] || [ "$1" = "-r" ]; then
    RAW_OUTPUT=true
    shift
fi

# Check if document ID is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Document ID is required${NC}"
    echo "Usage: $0 [OPTIONS] <document_id> [workspace] [qdrant_url] [api_key]"
    echo "Run '$0 --help' for more information"
    exit 1
fi

DOCUMENT_ID="$1"
WORKSPACE="${2:-${QDRANT_WORKSPACE:-_}}"
QDRANT_URL="${3:-${QDRANT_URL:-http://localhost:6333}}"
QDRANT_API_KEY="${4:-${QDRANT_API_KEY}}"

COLLECTION_NAME="lightrag_vdb_chunks"

echo -e "${GREEN}Querying Qdrant for document chunks...${NC}"
echo "Document ID: $DOCUMENT_ID"
echo "Workspace: $WORKSPACE"
echo "Qdrant URL: $QDRANT_URL"
echo "Collection: $COLLECTION_NAME"
echo ""

# Create a temporary Python script
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << 'PYTHON_SCRIPT'
import sys
import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

def main():
    document_id = sys.argv[1]
    workspace = sys.argv[2]
    qdrant_url = sys.argv[3]
    qdrant_api_key = sys.argv[4] if sys.argv[4] != "None" else None
    collection_name = sys.argv[5]
    raw_output = sys.argv[6] == "True"
    
    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # Check if collection exists
        if not client.collection_exists(collection_name):
            print(f"Error: Collection '{collection_name}' does not exist", file=sys.stderr)
            sys.exit(1)
        
        # Build filter for workspace and full_doc_id
        filter_conditions = [
            FieldCondition(
                key="workspace_id",
                match=MatchValue(value=workspace)
            ),
            FieldCondition(
                key="full_doc_id",
                match=MatchValue(value=document_id)
            )
        ]
        
        # Scroll through all matching points
        offset = None
        all_chunks = []
        batch_size = 100
        
        while True:
            result = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(must=filter_conditions),
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # Don't include vectors to reduce output size
            )
            
            points, next_offset = result
            
            if not points:
                break
            
            for point in points:
                chunk_data = {
                    "id": point.payload.get("id"),
                    "qdrant_id": str(point.id),
                    "full_doc_id": point.payload.get("full_doc_id"),
                    "content": point.payload.get("content"),
                    "file_path": point.payload.get("file_path"),
                    "workspace_id": point.payload.get("workspace_id"),
                    "created_at": point.payload.get("created_at"),
                    "chunk_order_index": point.payload.get("chunk_order_index"),
                    "tokens": point.payload.get("tokens"),
                    # Include any other payload fields
                    "all_payload": point.payload
                }
                all_chunks.append(chunk_data)
            
            if next_offset is None:
                break
            offset = next_offset
        
        if not all_chunks:
            print(f"No chunks found for document ID: {document_id}", file=sys.stderr)
            sys.exit(1)
        
        # Print results
        if raw_output:
            # Raw output: one JSON object per line
            for chunk in all_chunks:
                print(json.dumps(chunk, ensure_ascii=False))
        else:
            # Structured output with metadata
            output = {
                "document_id": document_id,
                "workspace": workspace,
                "collection": collection_name,
                "chunk_count": len(all_chunks),
                "chunks": all_chunks
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

# Check if qdrant-client is installed
if ! python3 -c "import qdrant_client" 2>/dev/null; then
    echo -e "${YELLOW}Warning: qdrant-client not found. Installing...${NC}"
    pip install -q qdrant-client
fi

# Run the Python script
python3 "$TEMP_SCRIPT" "$DOCUMENT_ID" "$WORKSPACE" "$QDRANT_URL" "${QDRANT_API_KEY:-None}" "$COLLECTION_NAME" "$RAW_OUTPUT"

# Clean up
rm -f "$TEMP_SCRIPT"

