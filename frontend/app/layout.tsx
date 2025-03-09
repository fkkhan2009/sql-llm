import './globals.css'
import { Inter } from 'next/font/google'
import Providers from './providers/Providers'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'LLM Database Query Interface',
  description: 'Natural language to SQL query interface with streaming responses',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
