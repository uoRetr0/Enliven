import { motion } from 'framer-motion'

export default function CharacterPill({ character, isSelected, onClick }) {
  return (
    <motion.button
      onClick={onClick}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={`
        px-4 py-2 rounded-full text-sm font-medium transition-all duration-200
        ${isSelected
          ? 'bg-accent text-white shadow-glow-sm'
          : 'bg-bg-card border border-white/10 text-text-secondary hover:border-accent/50 hover:text-text-primary'
        }
      `}
    >
      {character.name}
    </motion.button>
  )
}
