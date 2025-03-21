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
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const createWebSocketConnection = useCallback(() => {
    // Prevent multiple reconnection attempts
    if (isReconnecting.current) {
      console.log('Already attempting to reconnect, skipping...');
      return null;
    }

    isReconnecting.current = true;

    // Clear any existing reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close existing socket if it exists
    if (wsInstance.current && wsInstance.current.readyState !== WebSocket.CLOSED) {
      console.log('Closing existing WebSocket connection');
      wsInstance.current.close();
    }

    console.log('Creating new WebSocket connection');
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/api/enhanced/ws';
    console.log('WebSocket URL:', wsUrl);
    
    const ws = new WebSocket(wsUrl);

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
        
        // Attempt to reconnect after a delay
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect after connection closed...');
          createWebSocketConnection();
        }, 3000);
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

    // Add a ping interval to keep the connection alive
    const pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        console.log('Sending ping to keep connection alive');
        ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000); // Send ping every 30 seconds

    wsInstance.current = ws;
    setSocket(ws);
    
    // Return cleanup function
    return () => {
      clearInterval(pingInterval);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []); // Empty dependency array to ensure this function is created only once

  // Initialize WebSocket connection only once when the component mounts
  useEffect(() => {
    console.log('Initializing WebSocket connection');
    const cleanup = createWebSocketConnection();
    
    // Add event listener for page visibility changes
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        console.log('Page became visible, checking WebSocket connection');
        if (!wsInstance.current || wsInstance.current.readyState !== WebSocket.OPEN) {
          console.log('WebSocket not connected, reconnecting...');
          createWebSocketConnection();
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // Add event listener for online/offline status
    const handleOnline = () => {
      console.log('Browser is online, checking WebSocket connection');
      if (!wsInstance.current || wsInstance.current.readyState !== WebSocket.OPEN) {
        console.log('WebSocket not connected, reconnecting...');
        createWebSocketConnection();
      }
    };

    window.addEventListener('online', handleOnline);
    
    return () => {
      console.log('Cleaning up WebSocket connection');
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('online', handleOnline);
      
      if (cleanup) {
        cleanup();
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
