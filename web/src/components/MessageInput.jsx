import { useState } from 'react'
import { Send } from 'lucide-react'

export default function MessageInput({ onSend, disabled, compact }) {
  const [message, setMessage] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (message.trim() && !disabled) {
      onSend(message.trim())
      setMessage('')
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2">
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder="Type a message..."
        className={`
          flex-1 bg-bg-card border border-white/5 text-text-primary placeholder-text-muted
          focus:outline-none focus:border-accent/50 transition-colors disabled:opacity-50
          ${compact ? 'px-3 py-2 rounded-lg text-sm' : 'px-4 py-3 rounded-xl'}
        `}
      />

      <button
        type="submit"
        disabled={!message.trim() || disabled}
        className={`
          flex items-center justify-center transition-colors
          ${message.trim() && !disabled
            ? 'bg-accent text-white hover:bg-accent-dim'
            : 'bg-bg-card text-text-muted cursor-not-allowed'
          }
          ${compact ? 'p-2 rounded-lg' : 'p-3 rounded-xl'}
        `}
      >
        <Send className={compact ? 'w-4 h-4' : 'w-5 h-5'} />
      </button>
    </form>
  )
}
