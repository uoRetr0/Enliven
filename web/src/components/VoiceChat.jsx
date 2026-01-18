import { useState, useCallback } from 'react'
import { useConversation } from '@elevenlabs/react'
import { motion, AnimatePresence } from 'framer-motion'
import { Phone, PhoneOff, Mic, Volume2 } from 'lucide-react'

const API_BASE = 'http://localhost:8000'

export default function VoiceChat({ character, sessionId, onClose }) {
  const [error, setError] = useState(null)
  const [transcript, setTranscript] = useState([])
  const [isConnecting, setIsConnecting] = useState(false)

  const conversation = useConversation({
    onConnect: () => {
      console.log('Connected to ElevenLabs')
      setError(null)
      setIsConnecting(false)
    },
    onDisconnect: () => {
      console.log('Disconnected from ElevenLabs')
      setIsConnecting(false)
    },
    onMessage: (message) => {
      console.log('Message:', message)
      if (message.message) {
        setTranscript(prev => [...prev, {
          role: message.source === 'user' ? 'user' : 'assistant',
          text: message.message
        }])
      }
    },
    onError: (error) => {
      console.error('Conversation error:', error)
      setError(error.message || 'Connection error')
      setIsConnecting(false)
    }
  })

  const startConversation = useCallback(async () => {
    setIsConnecting(true)
    setError(null)

    try {
      // Request microphone permission first
      await navigator.mediaDevices.getUserMedia({ audio: true })

      // Get agent from backend (with caching)
      const response = await fetch(`${API_BASE}/api/create-agent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          character_id: character.id
        })
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to connect')
      }

      const { agent_id, signed_url } = await response.json()
      console.log('Agent ID:', agent_id)
      console.log('Signed URL:', signed_url)

      // Try connecting with agentId (more reliable)
      await conversation.startSession({ agentId: agent_id })
    } catch (err) {
      setIsConnecting(false)
      if (err.name === 'NotAllowedError') {
        setError('Microphone access denied')
      } else {
        setError(err.message || 'Failed to start conversation')
      }
    }
  }, [sessionId, character.id, conversation])

  const endConversation = useCallback(async () => {
    await conversation.endSession()
  }, [conversation])

  const isConnected = conversation.status === 'connected'

  return (
    <div className="flex flex-col h-full">
      {/* Status & Visual Feedback */}
      <div className="flex-1 flex flex-col items-center justify-center p-6">
        {/* Character Avatar with Speaking Indicator */}
        <div className="relative mb-6">
          <motion.div
            className={`
              w-24 h-24 rounded-full flex items-center justify-center text-3xl font-bold
              ${isConnected
                ? conversation.isSpeaking
                  ? 'bg-accent/30 text-accent'
                  : 'bg-green-500/20 text-green-400'
                : 'bg-bg-card text-text-muted'
              }
              border-2
              ${conversation.isSpeaking ? 'border-accent' : isConnected ? 'border-green-500' : 'border-white/20'}
            `}
            animate={conversation.isSpeaking ? {
              scale: [1, 1.05, 1],
              boxShadow: [
                '0 0 0 0 rgba(99, 102, 241, 0)',
                '0 0 0 15px rgba(99, 102, 241, 0.2)',
                '0 0 0 0 rgba(99, 102, 241, 0)'
              ]
            } : {}}
            transition={conversation.isSpeaking ? {
              duration: 1.5,
              repeat: Infinity,
              ease: 'easeInOut'
            } : {}}
          >
            {character.name.charAt(0).toUpperCase()}
          </motion.div>

          {/* Speaking/Listening indicator */}
          <AnimatePresence>
            {isConnected && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className={`
                  absolute -bottom-1 -right-1 p-1.5 rounded-full
                  ${conversation.isSpeaking ? 'bg-accent' : 'bg-green-500'}
                `}
              >
                {conversation.isSpeaking ? (
                  <Volume2 className="w-3 h-3 text-white" />
                ) : (
                  <Mic className="w-3 h-3 text-white" />
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Status Text */}
        <div className="text-center mb-6">
          {error ? (
            <p className="text-red-400 text-sm">{error}</p>
          ) : isConnecting ? (
            <p className="text-accent text-sm animate-pulse">Connecting...</p>
          ) : isConnected ? (
            <p className="text-sm">
              {conversation.isSpeaking ? (
                <span className="text-accent">{character.name} is speaking...</span>
              ) : (
                <span className="text-green-400">Listening to you...</span>
              )}
            </p>
          ) : (
            <p className="text-text-muted text-sm">Click to start voice chat</p>
          )}
        </div>

        {/* Waveform Visualization */}
        {isConnected && !conversation.isSpeaking && (
          <div className="flex items-center justify-center gap-1 h-8 mb-6">
            {[...Array(5)].map((_, i) => (
              <motion.div
                key={i}
                className="w-1 bg-green-400 rounded-full"
                animate={{
                  height: ['8px', '24px', '8px']
                }}
                transition={{
                  duration: 0.8,
                  repeat: Infinity,
                  delay: i * 0.1,
                  ease: 'easeInOut'
                }}
              />
            ))}
          </div>
        )}

        {/* Call Button */}
        <div className="flex gap-3">
          {!isConnected ? (
            <button
              onClick={startConversation}
              disabled={isConnecting}
              className={`
                flex items-center gap-2 px-6 py-3 rounded-full font-medium transition-all
                ${!isConnecting
                  ? 'bg-green-500 hover:bg-green-600 text-white'
                  : 'bg-bg-card border border-white/15 text-text-muted cursor-not-allowed'
                }
              `}
            >
              <Phone className="w-5 h-5" />
              {isConnecting ? 'Connecting...' : 'Start Voice Chat'}
            </button>
          ) : (
            <button
              onClick={endConversation}
              className="flex items-center gap-2 px-6 py-3 rounded-full font-medium bg-red-500 hover:bg-red-600 text-white transition-all"
            >
              <PhoneOff className="w-5 h-5" />
              End Call
            </button>
          )}
        </div>
      </div>

      {/* Transcript */}
      {transcript.length > 0 && (
        <div className="border-t border-white/15 p-4 max-h-40 overflow-y-auto">
          <p className="text-xs text-text-muted mb-2">Transcript</p>
          <div className="space-y-2">
            {transcript.slice(-5).map((msg, i) => (
              <div key={i} className={`text-xs ${msg.role === 'user' ? 'text-text-secondary' : 'text-accent'}`}>
                <span className="font-medium">{msg.role === 'user' ? 'You' : character.name}:</span> {msg.text}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer hint */}
      <div className="p-4 border-t border-white/15 text-center">
        <p className="text-xs text-text-muted">
          {isConnected
            ? 'Speak naturally - you can interrupt anytime'
            : 'Real-time voice conversation'
          }
        </p>
      </div>
    </div>
  )
}
