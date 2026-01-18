import { useRef, useEffect, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, GripVertical, Phone, MessageSquare } from 'lucide-react'
import ChatMessage from './ChatMessage'
import MessageInput from './MessageInput'
import VoiceChat from './VoiceChat'
import { getSessionId } from '../api/stubs'

const MIN_WIDTH = 280
const MAX_WIDTH = 480
const DEFAULT_WIDTH = 320

export default function ChatSidebar({
  isOpen,
  onClose,
  character,
  messages,
  onSend,
  isLoading,
  width,
  onWidthChange,
  audioEnabled,
  lastMessageId,
}) {
  const messagesEndRef = useRef(null)
  const [isResizing, setIsResizing] = useState(false)
  const [voiceMode, setVoiceMode] = useState(false)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleMouseDown = useCallback((e) => {
    e.preventDefault()
    setIsResizing(true)
  }, [])

  useEffect(() => {
    if (!isResizing) return

    const handleMouseMove = (e) => {
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, e.clientX))
      onWidthChange(newWidth)
    }

    const handleMouseUp = () => {
      setIsResizing(false)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizing, onWidthChange])

  return (
    <AnimatePresence>
      {isOpen && character && (
        <motion.div
          initial={{ x: '-100%' }}
          animate={{ x: 0 }}
          exit={{ x: '-100%' }}
          transition={{ duration: 0.15, ease: 'easeOut' }}
          style={{ width: width || DEFAULT_WIDTH }}
          className="fixed top-0 left-0 h-full bg-bg-primary border-r border-white/15 flex flex-col z-50"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-white/15">
            <div>
              <p className="text-xs text-text-muted mb-0.5">{voiceMode ? 'Voice chat with' : 'Chat with'}</p>
              <h3 className="font-semibold text-text-primary">{character.name}</h3>
            </div>
            <div className="flex items-center gap-1">
              {/* Mode toggle */}
              <button
                onClick={() => setVoiceMode(!voiceMode)}
                className={`
                  p-2 rounded-md transition-colors
                  ${voiceMode
                    ? 'text-accent bg-accent/10'
                    : 'text-text-muted hover:text-text-primary hover:bg-bg-card'
                  }
                `}
                title={voiceMode ? 'Switch to text chat' : 'Switch to voice chat'}
              >
                {voiceMode ? <MessageSquare className="w-4 h-4" /> : <Phone className="w-4 h-4" />}
              </button>
              <button
                onClick={onClose}
                className="p-2 rounded-md text-text-muted hover:text-text-primary hover:bg-bg-card transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Voice Chat Mode */}
          {voiceMode ? (
            <VoiceChat
              character={character}
              sessionId={getSessionId()}
              onClose={() => setVoiceMode(false)}
            />
          ) : (
            <>
              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center p-4">
                    <p className="text-text-muted text-sm mb-4">
                      Ask {character.name} anything about their story
                    </p>
                    <div className="space-y-2 w-full">
                      {[
                        "What's your story?",
                        "Tell me about yourself",
                      ].map((prompt) => (
                        <button
                          key={prompt}
                          onClick={() => onSend(prompt)}
                          className="w-full px-3 py-2 rounded-md bg-bg-card text-xs text-text-secondary hover:text-accent border border-white/10 hover:border-accent/30 transition-colors"
                        >
                          {prompt}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <>
                    {messages.map((message) => (
                      <ChatMessage
                        key={message.id}
                        message={message}
                        characterName={character.name}
                        autoPlay={audioEnabled && message.id === lastMessageId}
                      />
                    ))}

                    {isLoading && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.15 }}
                        className="flex items-start gap-2"
                      >
                        <div className="px-3 py-2 rounded-md bg-bg-card border border-white/10">
                          <div className="flex gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-text-muted typing-dot" />
                            <span className="w-1.5 h-1.5 rounded-full bg-text-muted typing-dot" />
                            <span className="w-1.5 h-1.5 rounded-full bg-text-muted typing-dot" />
                          </div>
                        </div>
                      </motion.div>
                    )}

                    <div ref={messagesEndRef} />
                  </>
                )}
              </div>

              {/* Input */}
              <div className="p-4 border-t border-white/15">
                <MessageInput onSend={onSend} disabled={isLoading} compact />
              </div>
            </>
          )}

          {/* Resize handle */}
          <div
            onMouseDown={handleMouseDown}
            className={`
              absolute top-0 right-0 w-1 h-full cursor-ew-resize group
              hover:bg-accent/50 transition-colors
              ${isResizing ? 'bg-accent/50' : 'bg-transparent'}
            `}
          >
            <div className="absolute top-1/2 right-0 -translate-y-1/2 translate-x-1/2 p-1 rounded bg-bg-card border border-white/15 opacity-0 group-hover:opacity-100 transition-opacity">
              <GripVertical className="w-3 h-3 text-text-muted" />
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
