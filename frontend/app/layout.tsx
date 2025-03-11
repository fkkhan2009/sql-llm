import './globals.css'
import { Inter } from 'next/font/google'
import Providers from './providers/Providers'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Database Query Chat',
  description: 'Natural language to SQL query interface with streaming responses',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-[#1e1e1e] text-[#f0f0f0]`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
