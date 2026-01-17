import { motion } from 'framer-motion'

export default function CharacterPill({ character, isSelected, onClick }) {
  return (
    <motion.button
      onClick={onClick}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={`
        px-3 py-1.5 rounded-sm text-sm font-medium transition-all duration-200
        ${isSelected
          ? 'bg-accent text-white shadow-glow-sm'
          : 'bg-bg-card border border-white/15 text-text-secondary hover:border-accent/50 hover:text-text-primary'
        }
      `}
    >
      {character.name}
    </motion.button>
  )
}
