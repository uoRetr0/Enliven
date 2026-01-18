import { motion } from 'framer-motion'
import { User, Baby, Dog } from 'lucide-react'

const getCharacterIcon = (character) => {
  const name = character.name?.toLowerCase() || ''
  const description = character.description?.toLowerCase() || ''
  const personality = character.personality?.toLowerCase() || ''
  const text = `${name} ${description} ${personality}`

  // Animals - all use Dog icon
  if (text.includes('wolf') || text.includes('dog') || text.includes('cat') || text.includes('bird') || text.includes('fish') || text.includes('rabbit') || text.includes('lamb') || text.includes('sheep') || text.includes('cow') || text.includes('bull') || text.includes('pig') || text.includes('goat') || text.includes('horse') || text.includes('camel') || text.includes('donkey') || text.includes('lion') || text.includes('tiger') || text.includes('bear') || text.includes('fox') || text.includes('deer') || text.includes('mouse') || text.includes('rat') || text.includes('elephant') || text.includes('monkey') || text.includes('snake') || text.includes('spider') || text.includes('turtle') || text.includes('frog') || text.includes('dragon')) return Dog
  
  // Humans
  if (text.includes('baby') || text.includes('infant') || text.includes('newborn') || text.includes('child') || text.includes('kid') || text.includes('toddler')) return Baby
  
  return User
}

export default function CharacterPill({ character, isSelected, onClick }) {
  const IconComponent = getCharacterIcon(character)
  
  return (
    <motion.button
      onClick={onClick}
      whileHover={{ scale: 1.05, y: -2 }}
      whileTap={{ scale: 0.95 }}
      className={`
        group relative px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-300 overflow-hidden
        ${isSelected
          ? 'bg-gradient-to-r from-accent to-accent-dim text-white shadow-lg shadow-accent/25'
          : 'bg-bg-card border border-white/20 text-text-secondary hover:border-accent/60 hover:text-text-primary hover:shadow-md'
        }
      `}
    >
      {/* Background glow effect */}
      <div className={`absolute inset-0 bg-gradient-to-r from-accent/10 to-accent-dim/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 ${isSelected ? 'opacity-100' : ''}`} />
      
      {/* Content */}
      <div className="relative flex items-center gap-2">
        <IconComponent className={`w-4 h-4 ${isSelected ? 'text-white' : 'text-text-muted group-hover:text-accent'} transition-colors duration-200`} />
        <span className="font-semibold">{character.name}</span>
        {character.type && (
          <span className={`text-xs px-1.5 py-0.5 rounded-full ${isSelected ? 'bg-white/20 text-white' : 'bg-accent/10 text-accent'}`}>
            {character.type}
          </span>
        )}
      </div>
    </motion.button>
  )
}
