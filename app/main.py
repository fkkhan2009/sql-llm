"""
Updated main application entry point with enhanced router
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

from app.config import settings
from app.dependencies import get_components  # Make sure components are initialized
from app.routers import base_router, workspace_router, conversation_router
from app.routers.enhanced import router as enhanced_router

# Create the FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version="0.2.0",
    description="Enhanced LLM QA Application with multi-step reasoning"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(base_router)
app.include_router(workspace_router)
app.include_router(conversation_router)
app.include_router(enhanced_router)  # Add the enhanced router

# Create a static directory for UI files if it doesn't exist
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

# Basic HTML template for the UI
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Database Query Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/lucide-react@0.263.1/dist/umd/lucide-react.development.js"></script>
</head>
<body class="bg-gray-100 min-h-screen">
    <div id="root"></div>
    <script src="/static/app.js"></script>
</body>
</html>
"""

# Route for the web UI
@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    return HTMLResponse(content=html_content)

# Mount the static files directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Make sure components are loaded at startup
@app.on_event("startup")
async def startup_event():
    """Load components on application startup"""
    get_components()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
