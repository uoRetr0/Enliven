import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Mic, Square } from 'lucide-react'

const API_BASE = 'http://localhost:8000'

// Audio detection settings (matching backend cli.py)
const SILENCE_THRESHOLD = 0.01
const SILENCE_DURATION = 1500
const MAX_RECORDING_TIME = 20000
const SAMPLE_RATE = 16000

export default function MessageInput({ onSend, disabled, compact }) {
  const [message, setMessage] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [micStatus, setMicStatus] = useState('idle')
  const [debugInfo, setDebugInfo] = useState('')
  const audioContextRef = useRef(null)
  const processorRef = useRef(null)
  const streamRef = useRef(null)
  const audioChunksRef = useRef([])
  const silenceStartRef = useRef(null)
  const speechDetectedRef = useRef(false)
  const maxTimeoutRef = useRef(null)
  const onSendRef = useRef(onSend)

  useEffect(() => {
    onSendRef.current = onSend
  }, [onSend])

  const stopRecording = useCallback(async () => {
    if (maxTimeoutRef.current) {
      clearTimeout(maxTimeoutRef.current)
      maxTimeoutRef.current = null
    }

    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current = null
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    if (audioContextRef.current) {
      await audioContextRef.current.close()
      audioContextRef.current = null
    }

    // Process recorded audio
    if (audioChunksRef.current.length === 0 || !speechDetectedRef.current) {
      setDebugInfo('No speech detected')
      setMicStatus('idle')
      setIsRecording(false)
      return
    }

    setMicStatus('processing')
    setDebugInfo('Transcribing...')

    try {
      // Combine all chunks into single Int16Array
      const totalLength = audioChunksRef.current.reduce((acc, chunk) => acc + chunk.length, 0)
      const combinedAudio = new Int16Array(totalLength)
      let offset = 0
      for (const chunk of audioChunksRef.current) {
        combinedAudio.set(chunk, offset)
        offset += chunk.length
      }

      // Convert to base64 (chunked to avoid stack overflow)
      const bytes = new Uint8Array(combinedAudio.buffer)
      let binary = ''
      const chunkSize = 8192
      for (let i = 0; i < bytes.length; i += chunkSize) {
        const chunk = bytes.subarray(i, i + chunkSize)
        binary += String.fromCharCode.apply(null, chunk)
      }
      const base64Audio = btoa(binary)

      const response = await fetch(`${API_BASE}/api/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio: base64Audio,
          sample_rate: SAMPLE_RATE,
          sample_width: 2  // 16-bit = 2 bytes
        })
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Transcription failed')
      }

      const data = await response.json()

      if (data.text) {
        setDebugInfo('')
        onSendRef.current(data.text)
      } else {
        setDebugInfo('No speech detected')
      }
    } catch (err) {
      setDebugInfo(err.message || 'Transcription error')
      setMicStatus('error')
    }

    setMicStatus('idle')
    setIsRecording(false)
  }, [])

  const startRecording = useCallback(async () => {
    try {
      setDebugInfo('Requesting mic...')
      setMicStatus('starting')

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }
      })

      streamRef.current = stream

      // Create audio context at target sample rate
      const audioContext = new AudioContext({ sampleRate: SAMPLE_RATE })
      audioContextRef.current = audioContext

      const source = audioContext.createMediaStreamSource(stream)

      // Use ScriptProcessor to capture raw PCM (deprecated but widely supported)
      const processor = audioContext.createScriptProcessor(4096, 1, 1)
      processorRef.current = processor

      audioChunksRef.current = []
      silenceStartRef.current = null
      speechDetectedRef.current = false

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0)

        // Calculate RMS for silence detection
        let sum = 0
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i]
        }
        const rms = Math.sqrt(sum / inputData.length)

        const now = Date.now()

        if (rms > SILENCE_THRESHOLD) {
          if (!speechDetectedRef.current) {
            speechDetectedRef.current = true
            setDebugInfo('Recording...')
          }
          silenceStartRef.current = null
        } else if (speechDetectedRef.current) {
          if (!silenceStartRef.current) {
            silenceStartRef.current = now
          } else if (now - silenceStartRef.current > SILENCE_DURATION) {
            setDebugInfo('Processing...')
            stopRecording()
            return
          }
        }

        // Convert float32 [-1, 1] to int16 [-32768, 32767]
        const int16Data = new Int16Array(inputData.length)
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]))
          int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
        }

        audioChunksRef.current.push(int16Data)
      }

      source.connect(processor)
      processor.connect(audioContext.destination)

      setMicStatus('recording')
      setDebugInfo('Listening...')
      setIsRecording(true)

      // Max recording timeout
      maxTimeoutRef.current = setTimeout(() => {
        setDebugInfo('Max time reached')
        stopRecording()
      }, MAX_RECORDING_TIME)

    } catch (err) {
      if (err.name === 'NotAllowedError') {
        setDebugInfo('Mic access denied')
      } else {
        setDebugInfo('Failed to start recording')
      }
      setMicStatus('error')
      setIsRecording(false)
    }
  }, [stopRecording])

  const toggleRecording = () => {
    if (disabled) return

    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }

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

  useEffect(() => {
    return () => {
      if (maxTimeoutRef.current) clearTimeout(maxTimeoutRef.current)
      if (processorRef.current) processorRef.current.disconnect()
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
      if (audioContextRef.current) audioContextRef.current.close()
    }
  }, [])

  return (
    <div className="space-y-1">
      {debugInfo && (
        <p className="text-xs text-accent px-1">{debugInfo}</p>
      )}

      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <button
          type="button"
          onClick={toggleRecording}
          disabled={disabled || micStatus === 'processing'}
          className={`
            flex items-center justify-center border transition-colors
            ${isRecording
              ? 'bg-red-500/20 border-red-500/50 text-red-400'
              : micStatus === 'error'
                ? 'bg-bg-card border-red-500/30 text-red-400/50'
                : micStatus === 'processing'
                  ? 'bg-bg-card border-accent/30 text-accent animate-pulse'
                  : disabled
                    ? 'bg-bg-card border-white/5 text-text-muted cursor-not-allowed'
                    : 'bg-bg-card border-white/10 text-text-secondary hover:text-accent hover:border-accent/30'
            }
            ${compact ? 'p-2 rounded-md' : 'p-3 rounded-lg'}
          `}
        >
          {isRecording ? (
            <Square className={compact ? 'w-4 h-4' : 'w-5 h-5'} />
          ) : (
            <Mic className={compact ? 'w-4 h-4' : 'w-5 h-5'} />
          )}
        </button>

        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled || isRecording || micStatus === 'processing'}
          placeholder={
            isRecording ? 'Listening...' :
            micStatus === 'processing' ? 'Transcribing...' :
            'Type or click mic...'
          }
          className={`
            flex-1 bg-bg-card border border-white/10 text-text-primary placeholder-text-muted
            focus:outline-none focus:border-accent/50 transition-colors disabled:opacity-50
            ${compact ? 'px-3 py-2 rounded-md text-sm' : 'px-4 py-3 rounded-lg'}
          `}
        />

        <button
          type="submit"
          disabled={!message.trim() || disabled}
          className={`
            flex items-center justify-center border transition-colors
            ${message.trim() && !disabled
              ? 'bg-accent border-accent text-white hover:bg-accent-dim'
              : 'bg-bg-card border-white/10 text-text-muted cursor-not-allowed'
            }
            ${compact ? 'p-2 rounded-md' : 'p-3 rounded-lg'}
          `}
        >
          <Send className={compact ? 'w-4 h-4' : 'w-5 h-5'} />
        </button>
      </form>
    </div>
  )
}
