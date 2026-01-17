import { motion } from 'framer-motion'
import { MessageSquare, Volume2, VolumeX } from 'lucide-react'

export default function Header({
  cost = 0,
  onToggleChat,
  isChatOpen,
  showChatToggle,
  audioEnabled,
  onToggleAudio,
}) {
  return (
    <header className="flex items-center justify-between mb-6">
      <h1 className="text-xl font-bold font-display gradient-text">
        ENLIVEN
      </h1>

      <div className="flex items-center gap-2">
        {/* Audio toggle */}
        <button
          onClick={onToggleAudio}
          className={`
            p-2 rounded-md border transition-colors
            ${audioEnabled
              ? 'bg-bg-card border-white/10 text-text-secondary hover:text-accent'
              : 'bg-bg-card border-white/10 text-text-muted'
            }
          `}
          title={audioEnabled ? 'Mute audio' : 'Enable audio'}
        >
          {audioEnabled ? (
            <Volume2 className="w-4 h-4" />
          ) : (
            <VolumeX className="w-4 h-4" />
          )}
        </button>

        {/* Cost badge */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-bg-card border border-white/10">
          <span className="text-xs text-text-muted">$</span>
          <span className="text-sm font-medium text-text-secondary">
            {cost.toFixed(2)}
          </span>
        </div>

        {/* Chat toggle */}
        {showChatToggle && (
          <motion.button
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.15 }}
            onClick={onToggleChat}
            className={`
              flex items-center gap-2 px-3 py-1.5 rounded-md border transition-colors
              ${isChatOpen
                ? 'bg-accent border-accent text-white'
                : 'bg-bg-card border-white/10 text-text-secondary hover:text-text-primary hover:border-white/20'
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
