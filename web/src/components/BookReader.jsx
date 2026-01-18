import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Upload, FileText, Loader2, Play, Pause, Volume2, RefreshCw } from 'lucide-react'
import CharacterPill from './CharacterPill'
import { getSessionId } from '../api/stubs'

export default function BookReader({
  bookText,
  characters,
  selectedCharacter,
  onExtract,
  onSelectCharacter,
  isLoading,
  hasExtracted,
  preloadedAudiobook,
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
  const [segmentDurations, setSegmentDurations] = useState([]) // Track actual durations
  const [totalDuration, setTotalDuration] = useState(0)
  const [elapsedBefore, setElapsedBefore] = useState(0) // Time elapsed before current segment
  const audioRef = useRef(null)

  // Load preloaded audiobook when available
  useEffect(() => {
    if (preloadedAudiobook && !audiobookData) {
      setAudiobookData(preloadedAudiobook)
    }
  }, [preloadedAudiobook])

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
    setSegmentDurations([])
    setTotalDuration(0)
    setElapsedBefore(0)

    setIsGeneratingAudiobook(true)

    try {
      const response = await fetch('http://localhost:8000/api/generate-audiobook', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: bookText, session_id: getSessionId() })
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
      // Update elapsed time before moving to next segment
      const currentDuration = audioRef.current?.duration || 0
      setElapsedBefore(prev => prev + currentDuration)
      setCurrentSegment(prev => prev + 1)
      setCurrentWordIndex(-1)
    } else {
      setIsPlaying(false)
      setCurrentSegment(0)
      setCurrentWordIndex(-1)
      setElapsedBefore(0)
    }
  }

  // Track segment duration when audio loads
  const handleLoadedMetadata = () => {
    if (!audioRef.current || !audiobookData) return
    const duration = audioRef.current.duration

    setSegmentDurations(prev => {
      const updated = [...prev]
      updated[currentSegment] = duration
      // Recalculate total duration
      const total = updated.reduce((sum, d) => sum + (d || 0), 0)
      // Estimate remaining segments based on average
      const knownCount = updated.filter(d => d > 0).length
      const avgDuration = knownCount > 0 ? total / knownCount : 3
      const estimatedTotal = total + (audiobookData.segments.length - knownCount) * avgDuration
      setTotalDuration(estimatedTotal)
      return updated
    })
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

  // Format text for better readability
  const formatBookText = (text) => {
    if (!text) return ''

    let formatted = text.trim()

    // Detect title: find where weird capitalization ends
    // Title has: ALL CAPS words (THE, LAMB), MiXeD caps (WoLf, tHE)
    // Prose starts: when we stop seeing weird caps
    const words = formatted.split(/\s+/)
    let lastWeirdCapsIndex = -1
    let charCount = 0

    const isWeirdCaps = (w) => {
      // ALL CAPS (2+ consecutive uppercase)
      if (/[A-Z]{2,}/.test(w)) return true
      // MiXeD caps (lowercase followed by uppercase)
      if (/[a-z][A-Z]/.test(w)) return true
      // Starts with lowercase but has uppercase later (tHE)
      if (/^[a-z].*[A-Z]/.test(w)) return true
      return false
    }

    for (let i = 0; i < Math.min(words.length, 20); i++) {
      const word = words[i]
      if (isWeirdCaps(word)) {
        lastWeirdCapsIndex = i
      }
      charCount += word.length + 1

      // Stop searching after we've gone 5 words past last weird caps
      if (i > lastWeirdCapsIndex + 5 && lastWeirdCapsIndex >= 0) {
        break
      }
    }

    // Title ends after the last weird caps word
    if (lastWeirdCapsIndex >= 0) {
      let titleEndPos = 0
      for (let i = 0; i <= lastWeirdCapsIndex; i++) {
        titleEndPos += words[i].length + 1
      }

      if (titleEndPos > 3 && titleEndPos < 150) {
        const rawTitle = formatted.slice(0, titleEndPos).trim()
        const restOfText = formatted.slice(titleEndPos).trim()

        // Convert title to Title Case
        const titleCase = rawTitle
          .toLowerCase()
          .split(' ')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ')

        formatted = titleCase + '\n\n' + restOfText
      }
    }

    // Add line breaks between speakers
    // Pattern: punctuation + closing quote + space + opening quote (new speaker)
    // e.g., ?' ' or .' ' or ,' '
    formatted = formatted.replace(/([.!?][\u0027\u0022\u2018\u2019\u201C\u201D])\s+([\u0027\u0022\u2018\u2019\u201C\u201D])/g, '$1\n\n$2')

    // Break after dialogue attribution ends and new dialogue starts
    // e.g., said Lambikin; 'text  or  said the Wolf, 'text
    formatted = formatted.replace(/((?:said|replied|asked|cried|called|shouted|whispered|answered|exclaimed|snarled|gasped|thought)\s+[\w\s]+[;,])\s*([\u0027\u0022\u2018\u2019\u201C\u201D])/gi, '$1\n\n$2')

    // Clean up multiple newlines
    formatted = formatted.replace(/\n{3,}/g, '\n\n')

    // Don't start with newline
    formatted = formatted.replace(/^\n+/, '')

    return formatted
  }

  // Upload state
  if (!hasExtracted) {
    // Loading state with skeleton
    if (isLoading) {
      return (
        <div className="space-y-4">
          <div className="text-center mb-4">
            <h2 className="text-xl font-bold font-display mb-1">
              Extracting characters...
            </h2>
            <p className="text-text-muted text-sm">
              Analyzing your text for character dialogues
            </p>
          </div>

          {/* Skeleton preview */}
          <div className="rounded-lg border border-white/15 bg-bg-secondary p-4 space-y-3">
            <div className="h-3 bg-bg-card rounded animate-pulse w-3/4" />
            <div className="h-3 bg-bg-card rounded animate-pulse w-full" />
            <div className="h-3 bg-bg-card rounded animate-pulse w-2/3" />
          </div>

          {/* Character pills skeleton */}
          <div className="space-y-2">
            <div className="h-3 bg-bg-card rounded animate-pulse w-20" />
            <div className="flex gap-2">
              <div className="h-7 bg-bg-card rounded-sm animate-pulse w-16" />
              <div className="h-7 bg-bg-card rounded-sm animate-pulse w-20" />
              <div className="h-7 bg-bg-card rounded-sm animate-pulse w-14" />
            </div>
          </div>

          <div className="flex justify-center pt-2">
            <div className="flex items-center gap-2 text-accent">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Processing...</span>
            </div>
          </div>
        </div>
      )
    }

    return (
      <div className="space-y-4">
        <div className="text-center mb-2">
          <h2 className="text-xl font-bold font-display mb-1">
            Bring your books to <span className="gradient-text">life</span>
          </h2>
          <p className="text-text-muted text-sm">
            Upload or paste text to extract characters
          </p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`
            rounded-lg border border-dashed transition-all duration-200
            ${isDragging
              ? 'border-accent bg-accent/5'
              : 'border-white/15 hover:border-white/20'
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

          <div className="p-4">
            {/* Compact file drop / browse row */}
            <div className="flex items-center gap-3 mb-3">
              <motion.div
                animate={{ y: isDragging ? -2 : 0 }}
                className="flex items-center justify-center w-10 h-10 rounded-sm bg-bg-card border border-white/15"
              >
                {fileName ? (
                  <FileText className="w-5 h-5 text-accent" />
                ) : (
                  <Upload className="w-5 h-5 text-text-muted" />
                )}
              </motion.div>

              <div className="flex-1 min-w-0">
                {fileName ? (
                  <p className="text-text-primary font-medium text-sm truncate">{fileName}</p>
                ) : (
                  <p className="text-text-secondary text-sm">
                    Drop file or{' '}
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="text-accent hover:underline"
                    >
                      browse
                    </button>
                    {' '}<span className="text-text-muted">(PDF, TXT)</span>
                  </p>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2 mb-3">
              <div className="flex-1 h-px bg-white/15" />
              <span className="text-text-muted text-xs">or paste text</span>
              <div className="flex-1 h-px bg-white/15" />
            </div>

            {/* Textarea with inline sample link */}
            <div className="space-y-1.5">
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste your book text here..."
                className="w-full h-48 px-3 py-2.5 rounded-sm bg-bg-card border border-white/15 text-text-primary placeholder-text-muted text-sm resize-none focus:outline-none focus:border-accent/50 transition-colors"
              />
              <div className="flex justify-between items-center">
                <button
                  onClick={loadSampleText}
                  className="text-xs text-text-muted hover:text-accent transition-colors"
                >
                  Try sample text
                </button>
                <button
                  onClick={handleExtract}
                  disabled={!text.trim()}
                  className={`
                    flex items-center gap-1.5 px-4 py-1.5 rounded-sm font-medium text-sm transition-all duration-200
                    ${text.trim()
                      ? 'bg-accent hover:bg-accent-dim text-white'
                      : 'bg-bg-card text-text-muted cursor-not-allowed border border-white/15'
                    }
                  `}
                >
                  Extract Characters
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    )
  }

  // Reader state - show book text and characters
  const globalWordIndex = getGlobalWordIndex()

  // Render text with word highlighting when audiobook is active
  const renderTextWithHighlighting = () => {
    const formattedText = formatBookText(bookText)

    if (!audiobookData) {
      // No audiobook - render plain paragraphs
      return formattedText.split('\n\n').map((paragraph, index) => {
        // Check if this is a title (first paragraph, short, title-case)
        const isTitle = index === 0 && paragraph.length < 100 && !paragraph.includes('.')
        return (
          <p
            key={index}
            className={`leading-[1.8] mb-6 last:mb-0 ${
              isTitle
                ? 'text-xl font-bold text-text-primary text-center mb-8'
                : 'text-text-primary/90'
            }`}
          >
            {paragraph}
          </p>
        )
      })
    }

    // Audiobook active - render with word highlighting
    let wordCounter = 0
    return formattedText.split('\n\n').map((paragraph, pIndex) => {
      const isTitle = pIndex === 0 && paragraph.length < 100 && !paragraph.includes('.')
      return (
        <p
          key={pIndex}
          className={`leading-[1.8] mb-6 last:mb-0 ${
            isTitle ? 'text-xl font-bold text-center mb-8' : ''
          }`}
        >
          {paragraph.split(/\s+/).map((word, wIndex) => {
            const currentWordCounter = wordCounter
            wordCounter++
            const isNext = currentWordCounter === globalWordIndex + 1 && globalWordIndex >= 0
            return (
              <span
                key={wIndex}
                className={`transition-all duration-200 ${
                  isTitle
                    ? isNext ? 'text-purple-400 font-medium' : 'text-text-primary'
                    : isNext ? 'text-purple-400 font-medium' : 'text-text-primary/70'
                }`}
              >
                {word}{' '}
              </span>
            )
          })}
        </p>
      )
    })
  }

  // Calculate total progress based on actual time
  const getTotalProgress = () => {
    if (!audiobookData || !audioRef.current || totalDuration === 0) return 0
    const currentTime = audioRef.current.currentTime || 0
    const elapsed = elapsedBefore + currentTime
    return Math.min((elapsed / totalDuration) * 100, 100)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Unified audiobook control bar */}
      <div className="flex items-center gap-2 mb-3 p-2 rounded-sm bg-bg-card border border-white/15">
        {!audiobookData ? (
          <button
            onClick={handleGenerateAudiobook}
            disabled={isGeneratingAudiobook || !bookText.trim()}
            className={`
              flex items-center gap-2 px-3 py-1.5 rounded-sm text-sm font-medium transition-all
              ${isGeneratingAudiobook || !bookText.trim()
                ? 'text-text-muted cursor-not-allowed'
                : 'text-text-primary hover:text-accent'
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
          <>
            {/* Play/Pause button */}
            <button
              onClick={togglePlayPause}
              className="flex items-center justify-center w-8 h-8 rounded-sm bg-accent hover:bg-accent-dim text-white transition-all"
            >
              {isPlaying ? (
                <Pause className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4 ml-0.5" />
              )}
            </button>

            {/* Progress bar */}
            <div className="flex-1 mx-2">
              <div className="h-1 bg-bg-secondary rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-accent"
                  initial={false}
                  animate={{ width: `${getTotalProgress()}%` }}
                  transition={{ duration: 0.1 }}
                />
              </div>
            </div>

            {/* Regenerate button */}
            <button
              onClick={handleGenerateAudiobook}
              disabled={isGeneratingAudiobook}
              className="flex items-center justify-center w-8 h-8 rounded-sm text-text-muted hover:text-text-primary hover:bg-bg-secondary transition-all"
              title="Regenerate audiobook"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isGeneratingAudiobook ? 'animate-spin' : ''}`} />
            </button>
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
          onLoadedMetadata={handleLoadedMetadata}
          onCanPlay={handleCanPlay}
          onTimeUpdate={handleTimeUpdate}
          onEnded={handleSegmentEnd}
        />
      )}

      {/* Book text area - improved reading typography */}
      <div className="flex-1 overflow-y-auto rounded-lg bg-bg-secondary/50 border border-white/15 p-6 mb-3">
        <div className="max-w-2xl mx-auto text-[15px]">
          {renderTextWithHighlighting()}
        </div>
      </div>

      {/* Character pills - compact */}
      {characters.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-xs text-text-muted font-medium">Characters</p>
          <div className="flex flex-wrap gap-1.5">
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
