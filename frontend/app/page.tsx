'use client'

import QueryInterface from './components/QueryInterface'

export default function Home() {
  return (
    <main className="min-h-screen p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Database Query Interface</h1>
        <QueryInterface />
      </div>
    </main>
  )
}
