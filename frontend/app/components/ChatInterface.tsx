import { useState, useEffect, useRef, useCallback } from 'react';
import { useWebSocket } from './WebSocketConnection';
import { ChatMessage, StreamResponseType, ProcessStep } from '../types';
import LoadingSpinner from './LoadingSpinner';

export default function ChatInterface() {
  const { socket, isConnected, error, reconnect } = useWebSocket();
  const [question, setQuestion] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [streamingContent, setStreamingContent] = useState<any[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentAssistantMessage, setCurrentAssistantMessage] = useState<ChatMessage | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageHandlerSet = useRef(false);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentAssistantMessage]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!socket || !isConnected || !question.trim()) return;

    // Add user message to chat
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: question,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);

    // Create initial assistant message
    const assistantMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      metadata: {
        steps: new Map(),
      },
    };
    setCurrentAssistantMessage(assistantMessage);

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
    setQuestion('');
  };

  // Handle WebSocket messages
  const handleMessage = useCallback((event: MessageEvent) => {
    const response = JSON.parse(event.data);
    console.log('Received WebSocket message:', response);
    
    // Handle error messages
    if (response.error) {
      setIsProcessing(false);
      setCurrentAssistantMessage(current => {
        if (current) {
          const updatedMessage = { ...current };
          updatedMessage.content = `Error: ${response.error}`;
          setMessages(prev => [...prev, updatedMessage]);
          return null;
        }
        return current;
      });
      return;
    }

    // Handle normal stream messages
    if (response.type !== StreamResponseType.QUERY_COMPLETE) {
      setStreamingContent(prev => [...prev, response]);
      
      // Handle completion through EXPLANATION step
      if (response.type === StreamResponseType.STEP_END && 
          response.step === ProcessStep.EXPLANATION) {
        finishProcessing();
      }
    }
    
    // Only handle QUERY_COMPLETE if we haven't already finished processing
    if (response.type === StreamResponseType.QUERY_COMPLETE && isProcessing) {
      finishProcessing();
    }
  }, [isProcessing]);

  // Function to handle finishing the processing state
  const finishProcessing = useCallback(() => {
    if (!isProcessing) return; // Prevent duplicate processing
    
    setIsProcessing(false);
    
    // Finalize the assistant message
    setCurrentAssistantMessage(current => {
      if (current) {
        const messageId = current.id; // Store the ID
        setMessages(prev => {
          // Check if message with this ID already exists
          if (prev.some(msg => msg.id === messageId)) {
            return prev;
          }
          return [...prev, current];
        });
        return null;
      }
      return current;
    });
  }, [isProcessing]);

  // Handle WebSocket close
  const handleClose = useCallback(() => {
    console.log('WebSocket connection closed');
    
    // Always reset processing state when connection closes
    if (isProcessing) {
      setIsProcessing(false);
      
      // If there's a current assistant message, finalize it
      setCurrentAssistantMessage(current => {
        if (current) {
          const updatedMessage = { ...current };
          // Only add error message if the message doesn't have content
          if (!updatedMessage.content.trim()) {
            updatedMessage.content = "I couldn't complete the response. Please try again.";
          }
          setMessages(prev => [...prev, updatedMessage]);
          return null;
        }
        return current;
      });
    }
  }, [isProcessing]);

  // Set up WebSocket event handlers
  useEffect(() => {
    if (!socket || messageHandlerSet.current) return;

    console.log('Setting up WebSocket message handlers');
    socket.onmessage = handleMessage;
    socket.onclose = handleClose;
    messageHandlerSet.current = true;

    return () => {
      console.log('Cleaning up WebSocket message handlers');
      if (socket) {
        socket.onmessage = null;
        socket.onclose = null;
      }
      messageHandlerSet.current = false;
    };
  }, [socket, handleMessage, handleClose]);

  // Reset message handler flag when socket changes
  useEffect(() => {
    messageHandlerSet.current = false;
  }, [socket]);

  // Process streaming content to update the current assistant message
  useEffect(() => {
    if (!currentAssistantMessage || streamingContent.length === 0) return;

    const updatedMessage = { ...currentAssistantMessage };
    let sql = '';
    let explanation = '';
    let results: any = null;
    const steps = new Map(updatedMessage.metadata?.steps || new Map());

    streamingContent.forEach((content) => {
      if (content.step && content.type) {
        // Update steps
        const currentStep = steps.get(content.step) || { 
          status: 'pending', 
          message: '', 
          llmOutput: '' 
        };

        switch (content.type) {
          case StreamResponseType.STEP_START:
            steps.set(content.step, { 
              ...currentStep,
              status: 'in-progress', 
              message: content.message 
            });
            break;
          case StreamResponseType.STEP_END:
            steps.set(content.step, { 
              ...currentStep,
              status: 'completed', 
              message: content.message 
            });
            break;
          case StreamResponseType.ERROR:
            steps.set(content.step, { 
              ...currentStep,
              status: 'error', 
              message: content.content 
            });
            break;
          case StreamResponseType.LLM_CHUNK:
            steps.set(content.step, {
              ...currentStep,
              status: currentStep.status || 'in-progress',
              llmOutput: (currentStep.llmOutput || '') + content.content
            });
            
            // Update specific sections
            if (content.step === ProcessStep.SQL) {
              sql += content.content;
            } else if (content.step === ProcessStep.EXPLANATION) {
              explanation += content.content;
            }
            break;
          case StreamResponseType.RESULT:
            if (content.step === ProcessStep.EXECUTION) {
              results = content.content;
            }
            break;
        }
      }
    });

    // Update the assistant message content with the explanation
    updatedMessage.content = explanation;
    updatedMessage.metadata = {
      ...updatedMessage.metadata,
      sql,
      results,
      steps,
    };

    setCurrentAssistantMessage(updatedMessage);
  }, [streamingContent, currentAssistantMessage]);

  return (
    <div className="flex flex-col h-[calc(100vh-200px)] max-w-4xl mx-auto">
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-[#2a2a2a] rounded-md mb-4">
        {messages.length === 0 && !currentAssistantMessage && (
          <div className="text-center text-gray-400 py-8">
            Ask a question to start the conversation
          </div>
        )}
        
        {messages.map((message) => (
          <ChatMessageComponent key={message.id} message={message} />
        ))}
        
        {currentAssistantMessage && (
          <>
            <ChatMessageComponent message={currentAssistantMessage} />
            <div className="flex items-center justify-center py-2">
              <LoadingSpinner />
              <span className="ml-2 text-sm text-gray-400">Processing...</span>
            </div>
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="mb-6">
        <div className="flex flex-col space-y-2">
          <div className="flex">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="flex-1 p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-[#8e8ea0] bg-[#3a3a3a] text-[#f0f0f0] border-[#555]"
              placeholder="Ask a question about your data..."
              disabled={isProcessing}
            />
            <button
              type="submit"
              disabled={!isConnected || isProcessing || !question.trim()}
              className="ml-2 bg-[#3a3a3a] text-[#f0f0f0] px-4 py-2 rounded-md hover:bg-[#4a4a4a] disabled:bg-[#2a2a2a] disabled:text-gray-500"
            >
              Send
            </button>
          </div>
          {error && (
            <div className="bg-[#3a3a3a] border border-red-700 text-[#f0f0f0] px-4 py-3 rounded flex justify-between items-center">
              <span>{error}</span>
              <button 
                onClick={reconnect}
                className="bg-[#555] text-[#f0f0f0] px-2 py-1 rounded text-sm hover:bg-[#666]"
              >
                Reconnect
              </button>
            </div>
          )}
        </div>
      </form>
    </div>
  );
}

interface ChatMessageComponentProps {
  message: ChatMessage;
}

function ChatMessageComponent({ message }: ChatMessageComponentProps) {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className={`p-4 rounded-lg ${message.role === 'user' ? 'bg-[#3a3a3a] ml-12' : 'bg-[#2a2a2a] mr-12 border border-[#444]'}`}>
      <div className="flex items-start">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center mr-2 ${
          message.role === 'user' ? 'bg-[#555] text-[#f0f0f0]' : 'bg-[#8e8ea0] text-[#f0f0f0]'
        }`}>
          {message.role === 'user' ? 'U' : 'A'}
        </div>
        <div className="flex-1">
          <div className="font-medium text-[#f0f0f0]">
            {message.role === 'user' ? 'You' : 'Assistant'}
            <span className="text-xs text-gray-400 ml-2">
              {message.timestamp.toLocaleTimeString()}
            </span>
          </div>
          <div className="mt-1 whitespace-pre-wrap text-[#f0f0f0]">{message.content}</div>
          
          {message.metadata && (message.metadata.sql || message.metadata.results) && (
            <div className="mt-2">
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="text-sm text-[#8e8ea0] hover:underline"
              >
                {showDetails ? 'Hide details' : 'Show details'}
              </button>
              
              {showDetails && (
                <div className="mt-2 space-y-3">
                  {message.metadata.sql && (
                    <div className="bg-[#333] p-3 rounded border border-[#444]">
                      <h4 className="text-sm font-medium mb-1 text-gray-300">Generated SQL</h4>
                      <pre className="text-xs bg-[#222] text-[#f0f0f0] p-2 rounded overflow-x-auto">
                        {message.metadata.sql}
                      </pre>
                    </div>
                  )}
                  
                  {message.metadata.results && (
                    <div className="bg-[#333] p-3 rounded border border-[#444]">
                      <h4 className="text-sm font-medium mb-1 text-gray-300">Query Results</h4>
                      <pre className="text-xs bg-[#222] text-[#f0f0f0] p-2 rounded overflow-x-auto">
                        {JSON.stringify(message.metadata.results, null, 2)}
                      </pre>
                    </div>
                  )}
                  
                  {message.metadata.steps && message.metadata.steps.size > 0 && (
                    <div className="bg-[#333] p-3 rounded border border-[#444]">
                      <h4 className="text-sm font-medium mb-1 text-gray-300">Process Steps</h4>
                      <div className="space-y-1">
                        {Array.from(message.metadata.steps.entries()).map(([step, { status, message }]) => (
                          <div key={step} className="text-xs flex items-center text-gray-300">
                            <span className="mr-1">
                              {status === 'error' ? '❌' : status === 'completed' ? '✅' : '⏳'}
                            </span>
                            <span className="font-medium">{step}</span>
                            {message && <span className="ml-1 text-gray-400">- {message}</span>}
                            </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 