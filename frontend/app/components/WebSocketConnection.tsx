import { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';

interface WebSocketContextType {
  socket: WebSocket | null;
  isConnected: boolean;
  error: string | null;
  reconnect: () => void;
}

const WebSocketContext = createContext<WebSocketContextType>({
  socket: null,
  isConnected: false,
  error: null,
  reconnect: () => {},
});

export const WebSocketProvider = ({ children }: { children: React.ReactNode }) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isReconnecting = useRef(false);
  const wsInstance = useRef<WebSocket | null>(null);

  const createWebSocketConnection = useCallback(() => {
    // Prevent multiple reconnection attempts
    if (isReconnecting.current) {
      console.log('Already attempting to reconnect, skipping...');
      return null;
    }

    isReconnecting.current = true;

    // Close existing socket if it exists
    if (wsInstance.current && wsInstance.current.readyState !== WebSocket.CLOSED) {
      console.log('Closing existing WebSocket connection');
      wsInstance.current.close();
    }

    console.log('Creating new WebSocket connection');
    const ws = new WebSocket(process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/api/enhanced/ws');

    ws.onopen = () => {
      console.log('WebSocket connection established');
      setIsConnected(true);
      setError(null);
      isReconnecting.current = false;
    };

    ws.onclose = (event) => {
      console.log('WebSocket connection closed', event);
      setIsConnected(false);
      
      if (!event.wasClean) {
        setError('WebSocket connection closed unexpectedly');
      } else {
        setError(null);
      }
      
      isReconnecting.current = false;
    };

    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      setError('WebSocket error occurred');
      isReconnecting.current = false;
    };

    wsInstance.current = ws;
    setSocket(ws);
    
    return ws;
  }, []); // Empty dependency array to ensure this function is created only once

  // Initialize WebSocket connection only once when the component mounts
  useEffect(() => {
    console.log('Initializing WebSocket connection');
    const ws = createWebSocketConnection();
    
    return () => {
      console.log('Cleaning up WebSocket connection');
      if (ws) {
        ws.close();
      }
      wsInstance.current = null;
    };
  }, [createWebSocketConnection]);

  // Reconnect function
  const reconnect = useCallback(() => {
    if (isReconnecting.current) {
      console.log('Already attempting to reconnect, skipping...');
      return;
    }
    
    console.log('Attempting to reconnect WebSocket...');
    createWebSocketConnection();
  }, [createWebSocketConnection]);

  return (
    <WebSocketContext.Provider value={{ socket, isConnected, error, reconnect }}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => useContext(WebSocketContext);
