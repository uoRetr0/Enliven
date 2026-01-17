// API client for Enliven backend

const API_BASE = 'http://localhost:8000'

// Session ID stored after character extraction
let sessionId = null

/**
 * Extract characters from book text
 * @param {string} text - Book text content
 * @returns {Promise<Array>} - Array of character objects
 */
export async function extractCharacters(text) {
  const response = await fetch(`${API_BASE}/api/extract-characters`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })

  if (!response.ok) {
    throw new Error('Failed to extract characters')
  }

  const data = await response.json()
  sessionId = data.session_id

  return data.characters
}

/**
 * Send a message to a character
 * @param {string} characterId - Character ID
 * @param {string} message - User message
 * @returns {Promise<{text: string, audioUrl: string|null}>}
 */
export async function sendMessage(characterId, message) {
  if (!sessionId) {
    throw new Error('No session. Extract characters first.')
  }

  const response = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      character_id: characterId,
      message,
    }),
  })

  if (!response.ok) {
    throw new Error('Failed to send message')
  }

  const data = await response.json()

  return {
    text: data.text,
    audioUrl: data.audio_url,
  }
}

/**
 * Get current session ID
 * @returns {string|null}
 */
export function getSessionId() {
  return sessionId
}

/**
 * Get current session costs
 * @returns {Promise<{llm: number, tts: number, total: number}>}
 */
export async function getCosts() {
  const params = sessionId ? `?session_id=${sessionId}` : ''
  const response = await fetch(`${API_BASE}/api/costs${params}`)

  if (!response.ok) {
    return { llm: 0, tts: 0, total: 0 }
  }

  return response.json()
}
