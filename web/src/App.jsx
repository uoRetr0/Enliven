import { useState } from 'react'
import Header from './components/Header'
import BookReader from './components/BookReader'
import ChatSidebar from './components/ChatSidebar'
import { extractCharacters, sendMessage, getCosts } from './api/stubs'

function App() {
  const [bookText, setBookText] = useState('')
  const [characters, setCharacters] = useState([])
  const [selectedCharacter, setSelectedCharacter] = useState(null)
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [sidebarWidth, setSidebarWidth] = useState(320)
  const [totalCost, setTotalCost] = useState(0)
  const [hasExtracted, setHasExtracted] = useState(false)
  const [audioEnabled, setAudioEnabled] = useState(true)

  const handleExtract = async (text) => {
    setIsLoading(true)
    setBookText(text)

    try {
      const chars = await extractCharacters(text)
      setCharacters(chars)
      setHasExtracted(true)
    } catch (error) {
      console.error('Failed to extract characters:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSelectCharacter = (character) => {
    setSelectedCharacter(character)
    setMessages([])
    setIsChatOpen(true)
  }

  const handleSendMessage = async (text) => {
    if (!text.trim() || !selectedCharacter) return

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: text,
    }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      const response = await sendMessage(selectedCharacter.id, text)

      const charMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.text,
        audioUrl: response.audioUrl,
      }
      setMessages(prev => [...prev, charMessage])

      // Fetch real costs from backend
      const costs = await getCosts()
      setTotalCost(costs.total)
    } catch (error) {
      console.error('Failed to send message:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleToggleChat = () => {
    setIsChatOpen(!isChatOpen)
  }

  return (
    <div className="min-h-screen bg-bg-primary relative overflow-hidden">
      {/* Background gradient */}
      <div className="fixed inset-0 bg-gradient-radial from-accent/5 via-transparent to-transparent pointer-events-none" />

      {/* Noise overlay */}
      <div className="noise-overlay" />

      {/* Chat sidebar - left side */}
      <ChatSidebar
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
        character={selectedCharacter}
        messages={messages}
        onSend={handleSendMessage}
        isLoading={isLoading && hasExtracted}
        width={sidebarWidth}
        onWidthChange={setSidebarWidth}
        audioEnabled={audioEnabled}
      />

      {/* Main content */}
      <div
        className="relative z-10 h-screen flex flex-col transition-all duration-150"
        style={{ marginLeft: isChatOpen ? `${sidebarWidth}px` : '0' }}
      >
        <div className="max-w-4xl mx-auto w-full px-6 py-6 flex flex-col h-full">
          <Header
            cost={totalCost}
            onToggleChat={handleToggleChat}
            isChatOpen={isChatOpen}
            showChatToggle={hasExtracted && selectedCharacter}
            audioEnabled={audioEnabled}
            onToggleAudio={() => setAudioEnabled(!audioEnabled)}
          />

          <div className="flex-1 overflow-hidden">
            <BookReader
              bookText={bookText}
              characters={characters}
              selectedCharacter={selectedCharacter}
              onExtract={handleExtract}
              onSelectCharacter={handleSelectCharacter}
              isLoading={isLoading && !hasExtracted}
              hasExtracted={hasExtracted}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
