import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Play, Pause } from 'lucide-react'

export default function ChatMessage({ message, characterName }) {
  const [isPlaying, setIsPlaying] = useState(false)
  const audioRef = useRef(null)
  const isUser = message.role === 'user'

  useEffect(() => {
    if (message.audioUrl && !audioRef.current) {
      audioRef.current = new Audio(message.audioUrl)
      audioRef.current.onended = () => setIsPlaying(false)
      audioRef.current.onerror = () => setIsPlaying(false)
    }

    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current = null
      }
    }
  }, [message.audioUrl])

  const handlePlayAudio = () => {
    if (!audioRef.current) return

    if (isPlaying) {
      audioRef.current.pause()
      setIsPlaying(false)
    } else {
      audioRef.current.play()
      setIsPlaying(true)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 5 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.15 }}
      className="space-y-1"
    >
      {/* Name label */}
      <p className={`text-xs font-medium ${isUser ? 'text-text-muted' : 'text-accent'}`}>
        {isUser ? 'You' : characterName}
      </p>

      {/* Message text */}
      <div
        className={`
          text-sm leading-relaxed
          ${isUser ? 'text-text-secondary' : 'text-text-primary'}
        `}
      >
        {message.content}
      </div>

      {/* Audio playback */}
      {!isUser && message.audioUrl && (
        <button
          onClick={handlePlayAudio}
          className="flex items-center gap-1.5 mt-2 text-xs text-text-muted hover:text-accent transition-colors"
        >
          {isPlaying ? (
            <>
              <Pause className="w-3 h-3" />
              <span>Playing...</span>
            </>
          ) : (
            <>
              <Play className="w-3 h-3" />
              <span>Play voice</span>
            </>
          )}
        </button>
      )}
    </motion.div>
  )
}
