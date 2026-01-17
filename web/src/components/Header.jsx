import { motion } from 'framer-motion'
import { MessageSquare } from 'lucide-react'

export default function Header({ cost = 0, onToggleChat, isChatOpen, showChatToggle }) {
  return (
    <header className="flex items-center justify-between mb-6">
      <h1 className="text-xl font-bold font-display gradient-text">
        ENLIVEN
      </h1>

      <div className="flex items-center gap-3">
        {/* Cost badge */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-bg-card border border-white/5">
          <span className="text-xs text-text-muted">$</span>
          <span className="text-sm font-medium text-text-secondary">
            {cost.toFixed(2)}
          </span>
        </div>

        {/* Chat toggle */}
        {showChatToggle && (
          <motion.button
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.15 }}
            onClick={onToggleChat}
            className={`
              flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors
              ${isChatOpen
                ? 'bg-accent text-white'
                : 'bg-bg-card border border-white/5 text-text-secondary hover:text-text-primary hover:border-white/10'
              }
            `}
          >
            <MessageSquare className="w-4 h-4" />
            <span className="text-sm font-medium">Chat</span>
          </motion.button>
        )}
      </div>
    </header>
  )
}
