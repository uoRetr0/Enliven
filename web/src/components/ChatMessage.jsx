import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Play, Volume2 } from 'lucide-react'

export default function ChatMessage({ message, characterName, autoPlay = false }) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [audioReady, setAudioReady] = useState(false)
  const audioRef = useRef(null)
  const hasAutoPlayedRef = useRef(false)
  const isUser = message.role === 'user'

  useEffect(() => {
    if (!message.audioUrl || isUser) return

    const audio = new Audio(message.audioUrl)
    audioRef.current = audio

    audio.oncanplaythrough = () => {
      setAudioReady(true)

      // Auto-play only once when audio is ready and autoPlay is enabled
      if (autoPlay && !hasAutoPlayedRef.current) {
        hasAutoPlayedRef.current = true
        audio.play().catch(() => {
          // Browser blocked autoplay - user will need to click play
        })
      }
    }

    audio.onplay = () => setIsPlaying(true)
    audio.onpause = () => setIsPlaying(false)
    audio.onended = () => setIsPlaying(false)
    audio.onerror = () => {
      setAudioReady(false)
      setIsPlaying(false)
    }

    return () => {
      audio.pause()
      audio.src = ''
      audioRef.current = null
    }
  }, [message.audioUrl, autoPlay, isUser])

  const handlePlayAudio = () => {
    if (!audioRef.current || !audioReady) return

    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.currentTime = 0
      audioRef.current.play().catch(() => {})
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
          disabled={!audioReady}
          className={`
            flex items-center gap-1.5 mt-2 text-xs transition-colors
            ${audioReady
              ? 'text-text-muted hover:text-accent'
              : 'text-text-muted/50 cursor-not-allowed'
            }
          `}
        >
          {isPlaying ? (
            <>
              <Volume2 className="w-3 h-3" />
              <span>Playing...</span>
            </>
          ) : (
            <>
              <Play className="w-3 h-3" />
              <span>{audioReady ? 'Play' : 'Loading...'}</span>
            </>
          )}
        </button>
      )}
    </motion.div>
  )
}
