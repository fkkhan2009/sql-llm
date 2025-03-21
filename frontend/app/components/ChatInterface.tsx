import { useState, useEffect, useRef, useCallback, memo } from 'react';
import { useWebSocket } from './WebSocketConnection';
import { ChatMessage, StreamResponseType, ProcessStep } from '../types';
import LoadingSpinner from './LoadingSpinner';
import { ArrowLeft, Stethoscope, Send } from 'lucide-react';

interface ChatMessageComponentProps {
  message: ChatMessage;
}

const ChatMessageComponent = memo(({ message }: ChatMessageComponentProps) => {
  const isUser = message.role === 'user';
  const [showDetails, setShowDetails] = useState(false);
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] rounded-lg p-4 ${
        isUser ? 'bg-gray-200 text-gray-800' : 'bg-gray-200 text-gray-800'
      }`}>
        <p className="whitespace-pre-wrap">{message.content}</p>
        
        {message.metadata && (message.metadata.sql || message.metadata.results || (message.metadata.steps && message.metadata.steps.size > 0)) && (
          <div className="mt-2">
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-sm text-gray-600 hover:underline"
            >
              {showDetails ? 'Hide details' : 'Show details'}
            </button>
            
            {showDetails && (
              <div className="mt-2 space-y-3">
                {message.metadata.sql && (
                  <div className="bg-gray-100 p-3 rounded border border-gray-300">
                    <h4 className="text-sm font-medium mb-1 text-gray-700">Generated SQL</h4>
                    <pre className="text-xs bg-white text-gray-800 p-2 rounded overflow-x-auto border border-gray-200">
                      {message.metadata.sql}
                    </pre>
                  </div>
                )}
                
                {message.metadata.results && (
                  <div className="bg-gray-100 p-3 rounded border border-gray-300">
                    <h4 className="text-sm font-medium mb-1 text-gray-700">Query Results</h4>
                    <pre className="text-xs bg-white text-gray-800 p-2 rounded overflow-x-auto border border-gray-200">
                      {JSON.stringify(message.metadata.results, null, 2)}
                    </pre>
                  </div>
                )}
                
                {message.metadata.steps && message.metadata.steps.size > 0 && (
                  <div className="bg-gray-100 p-3 rounded border border-gray-300">
                    <h4 className="text-sm font-medium mb-1 text-gray-700">Process Steps</h4>
                    <div className="space-y-1">
                      {Array.from(message.metadata.steps.entries()).map(([step, { status, message }]) => (
                        <div key={step} className="text-xs flex items-center text-gray-700">
                          <span className="mr-1">
                            {status === 'error' ? '❌' : status === 'completed' ? '✅' : '⏳'}
                          </span>
                          <span className="font-medium">{step}</span>
                          {message && <span className="ml-1 text-gray-500">- {message}</span>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        
        {/* Only show loading spinner for in-progress steps when not showing details */}
        {!showDetails && message.metadata?.steps && (
          <div className="mt-2 text-sm text-gray-600">
            {Array.from(message.metadata.steps.entries())
              .filter(([_, data]) => data.status === 'in-progress')
              .slice(0, 1) // Only show one loading spinner
              .map(([step, _]) => (
                <div key={step} className="mt-1 flex items-center">
                  <LoadingSpinner />
                  <span className="ml-2 text-xs text-gray-500">Processing...</span>
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
});

ChatMessageComponent.displayName = 'ChatMessageComponent';

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
      id: `user-${Date.now()}`,
      role: 'user',
      content: question,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);

    // Create initial assistant message
    const assistantMessage: ChatMessage = {
      id: `assistant-${Date.now()}`,
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
        if (!current) return null;
        const updatedMessage = { ...current };
        updatedMessage.content = `Error: ${response.error}`;
        setMessages(prev => [...prev, updatedMessage]);
        return null;
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
    if (!isProcessing) return;
    
    setIsProcessing(false);
    setCurrentAssistantMessage(current => {
      if (!current) return null;
      
      setMessages(prev => {
        const exists = prev.some(msg => msg.id === current.id);
        if (exists) return prev;
        return [...prev, { ...current }];
      });
      
      return null;
    });
  }, [isProcessing]);

  // Handle WebSocket close
  const handleClose = useCallback(() => {
    console.log('WebSocket connection closed');
    
    if (isProcessing) {
      setIsProcessing(false);
      
      setCurrentAssistantMessage(current => {
        if (!current) return null;
        
        const updatedMessage = { ...current };
        if (!updatedMessage.content.trim()) {
          updatedMessage.content = "I couldn't complete the response. Please try again.";
        }
        
        setMessages(prev => {
          const exists = prev.some(msg => msg.id === current.id);
          if (exists) return prev;
          return [...prev, updatedMessage];
        });
        
        return null;
      });
    }
  }, [isProcessing]);

  // Set up WebSocket event handlers
  useEffect(() => {
    if (!socket) return;
    
    console.log('Setting up WebSocket message handlers');
    
    const onMessage = (event: MessageEvent) => handleMessage(event);
    const onClose = () => handleClose();
    
    socket.addEventListener('message', onMessage);
    socket.addEventListener('close', onClose);
    messageHandlerSet.current = true;

    return () => {
      console.log('Cleaning up WebSocket message handlers');
      socket.removeEventListener('message', onMessage);
      socket.removeEventListener('close', onClose);
      messageHandlerSet.current = false;
    };
  }, [socket, handleMessage, handleClose]);

  // Process streaming content to update the current assistant message
  useEffect(() => {
    if (!currentAssistantMessage || streamingContent.length === 0) return;

    const lastContent = streamingContent[streamingContent.length - 1];
    if (!lastContent) return;

    setCurrentAssistantMessage(prevMessage => {
      if (!prevMessage) return null;

      const updatedMessage = { ...prevMessage };
      let sql = '';
      let explanation = '';
      let results: any = null;
      const steps = new Map(updatedMessage.metadata?.steps || new Map());

      streamingContent.forEach((content) => {
        if (content.step && content.type) {
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

      updatedMessage.content = explanation;
      updatedMessage.metadata = {
        ...updatedMessage.metadata,
        sql,
        results,
        steps,
      };

      return updatedMessage;
    });
  }, [streamingContent]);

  return (
    <div className="min-h-screen bg-white p-4 flex flex-col items-center">
      {/* Header */}
      <div className="w-full max-w-4xl">
        <button className="flex items-center text-gray-600 hover:text-gray-800 px-4 py-2 rounded-lg border border-gray-200">
          <ArrowLeft className="w-4 h-4 mr-2" />
          <span>Self Service</span>
        </button>
      </div>

      {/* Title */}
      <div className="flex items-center justify-center my-8">
        <h1 className="text-xl font-semibold text-gray-800">AI Assist</h1>
        <Stethoscope className="w-5 h-5 ml-2 text-gray-600" />
      </div>

      {/* Connection Status */}
      {!isConnected && (
        <div className="w-full max-w-4xl mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-yellow-800">
          <div className="flex justify-between items-center">
            <span>WebSocket disconnected. {error && `Error: ${error}`}</span>
            <button 
              onClick={reconnect}
              className="px-3 py-1 bg-yellow-100 hover:bg-yellow-200 rounded-md text-sm"
            >
              Reconnect
            </button>
          </div>
        </div>
      )}

      {/* Chat Area - Fixed height with scrollable content */}
      <div className="w-full max-w-4xl bg-white rounded-lg shadow-sm h-[600px] mb-12 p-4 border border-gray-200 flex flex-col">
        <div className="flex-1 overflow-y-auto pr-2">
          {messages.length === 0 && !currentAssistantMessage ? (
            <div className="text-center text-gray-500 mt-8">
              Ask a question to start the conversation
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message) => (
                <ChatMessageComponent key={`message-${message.id}`} message={message} />
              ))}
              {currentAssistantMessage && (
                <ChatMessageComponent key={`current-${currentAssistantMessage.id}`} message={currentAssistantMessage} />
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="relative mb-8 w-full max-w-4xl">
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your data..."
            className="w-full p-4 pr-12 rounded-lg border border-gray-200 bg-white text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={!isConnected || isProcessing}
          />
          <button
            type="submit"
            disabled={!isConnected || isProcessing || !question.trim()}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 disabled:opacity-50"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  );
}