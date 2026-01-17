import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Upload, FileText, Loader2, Play, Pause, Volume2, RefreshCw } from 'lucide-react'
import CharacterPill from './CharacterPill'

export default function BookReader({
  bookText,
  characters,
  selectedCharacter,
  onExtract,
  onSelectCharacter,
  isLoading,
  hasExtracted,
}) {
  const [text, setText] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [fileName, setFileName] = useState(null)
  const fileInputRef = useRef(null)

  // Audiobook state
  const [isPlaying, setIsPlaying] = useState(false)
  const [audiobookData, setAudiobookData] = useState(null)
  const [currentSegment, setCurrentSegment] = useState(0)
  const [currentWordIndex, setCurrentWordIndex] = useState(-1)
  const [isGeneratingAudiobook, setIsGeneratingAudiobook] = useState(false)
  const audioRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)

    const file = e.dataTransfer.files[0]
    if (file) {
      handleFile(file)
    }
  }

  const handleFile = async (file) => {
    setFileName(file.name)

    if (file.type === 'text/plain') {
      const content = await file.text()
      setText(content)
    } else if (file.type === 'application/pdf') {
      setText('[Extracting text from PDF...]')

      try {
        const formData = new FormData()
        formData.append('file', file)

        const response = await fetch('http://localhost:8000/api/extract-pdf', {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          const error = await response.json()
          throw new Error(error.detail || 'Failed to extract PDF')
        }

        const data = await response.json()
        setText(data.text)
      } catch (error) {
        console.error('PDF extraction failed:', error)
        setText(`[Failed to extract PDF: ${error.message}]`)
      }
    }
  }

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFile(file)
    }
  }

  const handleExtract = () => {
    if (text.trim()) {
      onExtract(text)
    }
  }

  const loadSampleText = () => {
    const sample = `"Will you buy my hair?" asked Della.

"I buy hair," said Madame Sofronie. "Take yer hat off and let's have a sight at the looks of it."

"Twenty dollars," said Madame Sofronie, lifting the mass with a practised hand.

"Give it to me quick," said Della.

The door opened and Jim stepped in. His eyes were fixed upon Della with an expression she could not read.

"Jim, darling," she cried, "don't look at me that way. I had my hair cut off and sold it. Say 'Merry Christmas!' Jim, and let's be happy."

"You've cut off your hair?" asked Jim.

"Cut it off and sold it," said Della. "Don't you like me just as well, anyhow?"

Jim drew a package from his overcoat pocket. "Dell," said he, "let's put our Christmas presents away. I sold the watch to get the money to buy your combs."`

    setText(sample)
    setFileName('gift-of-the-magi.txt')
  }

  // Audiobook functions
  const handleGenerateAudiobook = async () => {
    if (!bookText.trim()) return

    // Clear existing audiobook first
    setAudiobookData(null)
    setCurrentSegment(0)
    setCurrentWordIndex(-1)
    setIsPlaying(false)

    setIsGeneratingAudiobook(true)

    try {
      const response = await fetch('http://localhost:8000/api/generate-audiobook', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: bookText })
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to generate audiobook')
      }

      const data = await response.json()
      setAudiobookData(data)
      setCurrentSegment(0)
      setCurrentWordIndex(-1)
    } catch (error) {
      console.error('Failed to generate audiobook:', error)
      alert('Failed to generate audiobook: ' + error.message)
    } finally {
      setIsGeneratingAudiobook(false)
    }
  }

  const handleTimeUpdate = () => {
    if (!audiobookData || !audioRef.current) return
    const currentTime = audioRef.current.currentTime
    const segment = audiobookData.segments[currentSegment]
    if (!segment || !segment.words) return

    // Find current word based on time
    const wordIdx = segment.words.findIndex(
      (w, i, arr) => currentTime >= w.start &&
        (i === arr.length - 1 || currentTime < arr[i + 1].start)
    )
    setCurrentWordIndex(wordIdx)
  }

  const handleSegmentEnd = () => {
    if (!audiobookData) return
    if (currentSegment < audiobookData.segments.length - 1) {
      setCurrentSegment(prev => prev + 1)
      setCurrentWordIndex(-1)
    } else {
      setIsPlaying(false)
      setCurrentSegment(0)
      setCurrentWordIndex(-1)
    }
  }

  // When audio is ready and we're supposed to be playing, start playback
  // This handles segment transitions where we need to wait for new audio to load
  const handleCanPlay = () => {
    if (isPlaying && audioRef.current) {
      audioRef.current.play().catch(err => {
        console.error('Play on canplay failed:', err)
      })
    }
  }

  // Toggle play/pause - direct control from click handler (required for autoplay policy)
  const togglePlayPause = () => {
    console.log('togglePlayPause called', { audiobookData: !!audiobookData, audioRef: !!audioRef.current, isPlaying })

    if (!audiobookData) {
      console.log('No audiobook data')
      return
    }

    // Audio ref might not be set yet on first render, set state and let onCanPlay handle it
    if (!audioRef.current) {
      console.log('No audio ref, setting isPlaying true')
      setIsPlaying(true)
      return
    }

    if (isPlaying) {
      console.log('Pausing')
      audioRef.current.pause()
      setIsPlaying(false)
    } else {
      const segment = audiobookData.segments[currentSegment]
      console.log('Playing segment:', currentSegment, 'audio_base64 length:', segment?.audio_base64?.length || 0)
      console.log('Audio src:', audioRef.current.src?.substring(0, 100))
      // Must call play() directly in click handler for browser autoplay policy
      audioRef.current.play()
        .then(() => {
          console.log('Play succeeded')
          setIsPlaying(true)
        })
        .catch(err => {
          console.error('Play failed:', err)
          // Still set playing true so onCanPlay can retry when audio is ready
          setIsPlaying(true)
        })
    }
  }

  // Build word index map for highlighting across all segments
  const getGlobalWordIndex = () => {
    if (!audiobookData || currentWordIndex < 0) return -1
    let globalIndex = 0
    for (let i = 0; i < currentSegment; i++) {
      globalIndex += audiobookData.segments[i].words?.length || 0
    }
    return globalIndex + currentWordIndex
  }

  // Upload state
  if (!hasExtracted) {
    return (
      <div className="space-y-6">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold font-display mb-2">
            Bring your books to <span className="gradient-text">life</span>
          </h2>
          <p className="text-text-secondary">
            Upload a book or paste text to extract characters
          </p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`
            relative rounded-2xl border-2 border-dashed transition-all duration-300
            ${isDragging
              ? 'border-accent bg-accent/5 scale-[1.01]'
              : 'border-white/10 bg-bg-secondary hover:border-white/20'
            }
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.pdf"
            onChange={handleFileSelect}
            className="hidden"
          />

          <div className="p-6">
            <div className="text-center mb-4">
              <motion.div
                animate={{ y: isDragging ? -5 : 0 }}
                className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-bg-card mb-3"
              >
                {fileName ? (
                  <FileText className="w-6 h-6 text-accent" />
                ) : (
                  <Upload className="w-6 h-6 text-text-muted" />
                )}
              </motion.div>

              {fileName ? (
                <p className="text-text-primary font-medium text-sm">{fileName}</p>
              ) : (
                <>
                  <p className="text-text-primary font-medium text-sm mb-1">
                    Drop your book here
                  </p>
                  <p className="text-text-muted text-xs">
                    or{' '}
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="text-accent hover:underline"
                    >
                      browse files
                    </button>
                    {' '}(PDF, TXT)
                  </p>
                </>
              )}
            </div>

            <div className="flex items-center gap-3 mb-4">
              <div className="flex-1 h-px bg-white/10" />
              <span className="text-text-muted text-xs">or paste text</span>
              <div className="flex-1 h-px bg-white/10" />
            </div>

            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your book text here..."
              className="w-full h-40 px-4 py-3 rounded-xl bg-bg-card border border-white/5 text-text-primary placeholder-text-muted text-sm resize-none focus:outline-none focus:border-accent/50 transition-colors"
            />

            <div className="flex justify-center mt-2">
              <button
                onClick={loadSampleText}
                className="text-xs text-text-muted hover:text-accent transition-colors"
              >
                Load sample text
              </button>
            </div>
          </div>
        </motion.div>

        <div className="flex justify-center">
          <button
            onClick={handleExtract}
            disabled={!text.trim() || isLoading}
            className={`
              flex items-center gap-2 px-6 py-2.5 rounded-xl font-medium text-sm transition-all duration-300
              ${text.trim() && !isLoading
                ? 'bg-accent hover:bg-accent-dim text-white shadow-glow-sm'
                : 'bg-bg-card text-text-muted cursor-not-allowed'
              }
            `}
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Extracting...</span>
              </>
            ) : (
              <span>Extract Characters</span>
            )}
          </button>
        </div>
      </div>
    )
  }

  // Reader state - show book text and characters
  const globalWordIndex = getGlobalWordIndex()

  // Render text with word highlighting when audiobook is active
  const renderTextWithHighlighting = () => {
    if (!audiobookData) {
      // No audiobook - render plain paragraphs
      return bookText.split('\n\n').map((paragraph, index) => (
        <p
          key={index}
          className="text-text-primary leading-relaxed mb-4 last:mb-0"
        >
          {paragraph}
        </p>
      ))
    }

    // Audiobook active - render with word highlighting
    let wordCounter = 0
    return bookText.split('\n\n').map((paragraph, pIndex) => (
      <p
        key={pIndex}
        className="text-text-primary leading-relaxed mb-4 last:mb-0"
      >
        {paragraph.split(/\s+/).map((word, wIndex) => {
          const currentWordCounter = wordCounter
          wordCounter++
          return (
            <span
              key={wIndex}
              className={`transition-colors duration-150 ${
                currentWordCounter === globalWordIndex
                  ? 'bg-accent/30 rounded px-0.5'
                  : ''
              }`}
            >
              {word}{' '}
            </span>
          )
        })}
      </p>
    ))
  }

  return (
    <div className="flex flex-col h-full">
      {/* Audiobook controls */}
      <div className="flex items-center gap-3 mb-4">
        {!audiobookData ? (
          <button
            onClick={handleGenerateAudiobook}
            disabled={isGeneratingAudiobook || !bookText.trim()}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all
              ${isGeneratingAudiobook || !bookText.trim()
                ? 'bg-bg-card text-text-muted cursor-not-allowed'
                : 'bg-bg-card hover:bg-bg-secondary text-text-primary border border-white/10 hover:border-white/20'
              }
            `}
          >
            {isGeneratingAudiobook ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Generating...</span>
              </>
            ) : (
              <>
                <Volume2 className="w-4 h-4" />
                <span>Generate Audiobook</span>
              </>
            )}
          </button>
        ) : (
          <button
            onClick={togglePlayPause}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium bg-accent hover:bg-accent-dim text-white transition-all"
          >
            {isPlaying ? (
              <>
                <Pause className="w-4 h-4" />
                <span>Pause</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Play</span>
              </>
            )}
          </button>
        )}

        {audiobookData && (
          <>
            <button
              onClick={handleGenerateAudiobook}
              disabled={isGeneratingAudiobook}
              className="flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium bg-bg-card hover:bg-bg-secondary text-text-secondary border border-white/10 hover:border-white/20 transition-all"
              title="Regenerate audiobook"
            >
              <RefreshCw className={`w-4 h-4 ${isGeneratingAudiobook ? 'animate-spin' : ''}`} />
            </button>
            <span className="text-xs text-text-muted">
              Segment {currentSegment + 1} / {audiobookData.segments.length}
            </span>
          </>
        )}
      </div>

      {/* Hidden audio element */}
      {audiobookData && (
        <audio
          ref={audioRef}
          src={
            audiobookData.segments[currentSegment]?.audio_base64
              ? `data:audio/mpeg;base64,${audiobookData.segments[currentSegment].audio_base64}`
              : undefined
          }
          onCanPlay={handleCanPlay}
          onTimeUpdate={handleTimeUpdate}
          onEnded={handleSegmentEnd}
        />
      )}

      {/* Book text area */}
      <div className="flex-1 overflow-y-auto rounded-2xl bg-bg-secondary border border-white/5 p-6 mb-4">
        <div className="prose prose-invert max-w-none">
          {renderTextWithHighlighting()}
        </div>
      </div>

      {/* Character pills */}
      {characters.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-text-muted">Characters in this book:</p>
          <div className="flex flex-wrap gap-2">
            {characters.map((character) => (
              <CharacterPill
                key={character.id}
                character={character}
                isSelected={selectedCharacter?.id === character.id}
                onClick={() => onSelectCharacter(character)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
