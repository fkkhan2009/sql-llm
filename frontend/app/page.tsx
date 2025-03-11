'use client'

import ChatInterface from './components/ChatInterface'

export default function Home() {
  return (
    <main className="min-h-screen bg-[#1e1e1e] text-[#f0f0f0]">
      <div className="max-w-6xl mx-auto p-4 md:p-8">
        <h1 className="text-3xl font-bold mb-8 text-[#f0f0f0]">Database Query Chat</h1>
        <ChatInterface />
      </div>
    </main>
  )
}
