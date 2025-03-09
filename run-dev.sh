#!/bin/bash
# Run backend
uvicorn app.main:app --reload --port 8000 & 
# Run frontend
cd frontend && npm run dev &
# Wait for both processes
wait