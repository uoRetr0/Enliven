import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { Upload, FileText, Loader2 } from 'lucide-react'
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
      setText(`[PDF uploaded: ${file.name}]\n\nPDF text extraction will be handled by the backend.`)
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
    const sample = `The great hall of Hogwarts was filled with the chatter of students. Harry Potter sat between Ron Weasley and Hermione Granger at the Gryffindor table.

"I can't believe Snape gave us that much homework," Ron complained, his red hair falling into his eyes. "It's like he wants us to fail."

"If you'd actually paid attention in class, Ronald, you'd know it's not that difficult," Hermione said primly, already pulling out her books.

Harry just smiled. After years of living with the Dursleys, any amount of homework was worth it to be here, at Hogwarts, with his friends. He still remembered the day Hagrid had burst into that hut on the rock and told him he was a wizard.

Professor Dumbledore stood at the head table, his long silver beard catching the candlelight. His blue eyes twinkled behind half-moon spectacles as he watched the students with a knowing smile.`

    setText(sample)
    setFileName('sample-text.txt')
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
  return (
    <div className="flex flex-col h-full">
      {/* Book text area */}
      <div className="flex-1 overflow-y-auto rounded-2xl bg-bg-secondary border border-white/5 p-6 mb-4">
        <div className="prose prose-invert max-w-none">
          {bookText.split('\n\n').map((paragraph, index) => (
            <p
              key={index}
              className="text-text-primary leading-relaxed mb-4 last:mb-0"
            >
              {paragraph}
            </p>
          ))}
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
