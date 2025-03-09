import { useState, useEffect } from 'react';
import { useWebSocket } from './WebSocketConnection';
import QueryResults from './QueryResults';
import ProcessSteps from './ProcessSteps';

export default function QueryInterface() {
  const { socket, isConnected, error } = useWebSocket();
  const [question, setQuestion] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [streamingContent, setStreamingContent] = useState<any[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!socket || !isConnected) return;

    setIsProcessing(true);
    setStreamingContent([]);

    const query = {
      question,
      workspace_id: null,
      include_docs: true,
      save_as_view: false,
      temperature: 0.3,
      use_progressive_building: true,
    };

    socket.send(JSON.stringify(query));
  };

  useEffect(() => {
    if (!socket) return;

    socket.onmessage = (event) => {
      const response = JSON.parse(event.data);
      console.log('Received WebSocket message:', response);
      setStreamingContent((prev) => [...prev, response]);
    };
  }, [socket]);

  return (
    <div className="max-w-4xl mx-auto p-4">
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="flex flex-col space-y-2">
          <label htmlFor="question" className="text-sm font-medium">
            Ask a question about your data
          </label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            className="p-2 border rounded-md"
            rows={3}
            placeholder="Enter your question..."
          />
          <button
            type="submit"
            disabled={!isConnected || isProcessing}
            className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-400"
          >
            {isProcessing ? 'Processing...' : 'Submit'}
          </button>
        </div>
      </form>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      <ProcessSteps streamingContent={streamingContent} />
      <QueryResults streamingContent={streamingContent} />
    </div>
  );
}
