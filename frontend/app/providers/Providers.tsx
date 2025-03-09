'use client'

import { WebSocketProvider } from '../components/WebSocketConnection'

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <WebSocketProvider>
      {children}
    </WebSocketProvider>
  )
}
