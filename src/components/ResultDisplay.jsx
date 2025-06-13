import React from 'react'
import { motion } from 'framer-motion'
import { RefreshCw, Download, Share2, Eye, Heart, Frown, Smile, Meh, Angry, Sunrise as Surprise } from 'lucide-react'

const emotionIcons = {
  happy: Smile,
  sad: Frown,
  angry: Angry,
  surprise: Surprise,
  neutral: Meh,
  fear: Eye,
  disgust: Frown
}

const ResultDisplay = ({ results, originalImage, onReset }) => {
  const downloadResult = () => {
    const link = document.createElement('a')
    link.href = results.image
    link.download = 'face-detection-result.jpg'
    link.click()
  }

  if (results.error) {
    return (
      <motion.div
        className="text-center py-16"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
      >
        <motion.div
          className="glass rounded-3xl p-12 max-w-2xl mx-auto"
          animate={{ 
            boxShadow: [
              "0 0 20px rgba(255, 0, 0, 0.3)",
              "0 0 40px rgba(255, 0, 0, 0.5)",
              "0 0 20px rgba(255, 0, 0, 0.3)"
            ]
          }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <div className="text-6xl mb-6">⚠️</div>
          <h3 className="text-2xl font-bold text-red-400 mb-4">Processing Error</h3>
          <p className="text-gray-300 mb-8">{results.error}</p>
          <motion.button
            onClick={onReset}
            className="px-8 py-4 bg-gradient-to-r from-red-500 to-red-600 rounded-xl font-semibold text-white flex items-center gap-3 mx-auto hover:shadow-lg hover:shadow-red-500/25 transition-all duration-300"
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
          >
            <RefreshCw size={20} />
            Try Again
          </motion.button>
        </motion.div>
      </motion.div>
    )
  }

  return (
    <motion.div
      className="space-y-8"
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      {/* Action Bar */}
      <motion.div 
        className="flex justify-center gap-4"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <motion.button
          onClick={downloadResult}
          className="px-6 py-3 glass rounded-xl flex items-center gap-2 hover:bg-white/10 transition-all duration-300"
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
        >
          <Download size={18} />
          Download
        </motion.button>
        
        <motion.button
          className="px-6 py-3 glass rounded-xl flex items-center gap-2 hover:bg-white/10 transition-all duration-300"
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
        >
          <Share2 size={18} />
          Share
        </motion.button>
        
        <motion.button
          onClick={onReset}
          className="px-6 py-3 bg-gradient-to-r from-neon-blue to-neon-purple rounded-xl flex items-center gap-2 font-semibold text-white hover:shadow-lg hover:shadow-neon-blue/25 transition-all duration-300"
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
        >
          <RefreshCw size={18} />
          Analyze New Image
        </motion.button>
      </motion.div>

      {/* Image Comparison */}
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Original Image */}
        <motion.div
          className="glass rounded-3xl p-6"
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <Eye className="text-neon-blue" />
            Original Image
          </h3>
          <div className="relative rounded-2xl overflow-hidden">
            <img 
              src={originalImage} 
              alt="Original" 
              className="w-full h-auto"
            />
            <motion.div
              className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            />
          </div>
        </motion.div>

        {/* Result Image */}
        <motion.div
          className="glass rounded-3xl p-6"
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <Eye className="text-neon-green" />
            Detection Result
          </h3>
          <div className="relative rounded-2xl overflow-hidden">
            <img 
              src={results.image} 
              alt="Result" 
              className="w-full h-auto"
            />
            <motion.div
              className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
            />
          </div>
        </motion.div>
      </div>

      {/* Face Analysis Results */}
      {results.faces && results.faces.length > 0 && (
        <motion.div
          className="glass rounded-3xl p-8"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <h3 className="text-2xl font-bold gradient-text mb-6 text-center">
            Face Analysis Results
          </h3>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {results.faces.map((face, index) => {
              const EmotionIcon = emotionIcons[face.emotion?.toLowerCase()] || Heart
              
              return (
                <motion.div
                  key={index}
                  className="hologram rounded-2xl p-6 border border-white/10"
                  initial={{ opacity: 0, scale: 0.8, rotateY: -90 }}
                  animate={{ opacity: 1, scale: 1, rotateY: 0 }}
                  transition={{ 
                    delay: 0.6 + index * 0.1,
                    type: "spring",
                    stiffness: 100
                  }}
                  whileHover={{ 
                    scale: 1.05,
                    rotateY: 5,
                    transition: { duration: 0.2 }
                  }}
                >
                  <div className="text-center mb-4">
                    <motion.div
                      className="w-16 h-16 mx-auto mb-3 rounded-full bg-gradient-to-r from-neon-blue to-neon-purple flex items-center justify-center"
                      animate={{ 
                        rotate: [0, 360],
                        scale: [1, 1.1, 1]
                      }}
                      transition={{ 
                        duration: 4,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    >
                      <span className="text-2xl font-bold text-white">
                        {index + 1}
                      </span>
                    </motion.div>
                    <h4 className="text-lg font-semibold text-white">
                      Face {index + 1}
                    </h4>
                  </div>

                  <div className="space-y-4">
                    {face.emotion && (
                      <motion.div 
                        className="flex items-center justify-between p-3 rounded-xl bg-white/5"
                        whileHover={{ x: 5 }}
                      >
                        <div className="flex items-center gap-3">
                          <EmotionIcon className="text-neon-pink" size={20} />
                          <span className="text-gray-300">Emotion</span>
                        </div>
                        <span className="text-neon-pink font-semibold capitalize">
                          {face.emotion}
                        </span>
                      </motion.div>
                    )}

                    {face.age && (
                      <motion.div 
                        className="flex items-center justify-between p-3 rounded-xl bg-white/5"
                        whileHover={{ x: 5 }}
                      >
                        <div className="flex items-center gap-3">
                          <span className="text-neon-blue text-lg">🎂</span>
                          <span className="text-gray-300">Age</span>
                        </div>
                        <span className="text-neon-blue font-semibold">
                          {Math.round(face.age)} years
                        </span>
                      </motion.div>
                    )}

                    {face.gender && (
                      <motion.div 
                        className="flex items-center justify-between p-3 rounded-xl bg-white/5"
                        whileHover={{ x: 5 }}
                      >
                        <div className="flex items-center gap-3">
                          <span className="text-neon-purple text-lg">
                            {face.gender.toLowerCase() === 'man' ? '👨' : '👩'}
                          </span>
                          <span className="text-gray-300">Gender</span>
                        </div>
                        <span className="text-neon-purple font-semibold capitalize">
                          {face.gender}
                        </span>
                      </motion.div>
                    )}

                    <motion.div 
                      className="flex items-center justify-between p-3 rounded-xl bg-white/5"
                      whileHover={{ x: 5 }}
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-neon-green text-lg">🎯</span>
                        <span className="text-gray-300">Confidence</span>
                      </div>
                      <span className="text-neon-green font-semibold">
                        {(face.confidence * 100).toFixed(1)}%
                      </span>
                    </motion.div>
                  </div>
                </motion.div>
              )
            })}
          </div>

          {/* Summary */}
          <motion.div
            className="mt-8 text-center p-6 rounded-2xl bg-gradient-to-r from-neon-blue/10 to-neon-purple/10 border border-white/10"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
          >
            <p className="text-lg text-gray-300">
              Successfully detected <span className="text-neon-blue font-bold">{results.faces.length}</span> face{results.faces.length !== 1 ? 's' : ''} with advanced AI analysis
            </p>
          </motion.div>
        </motion.div>
      )}

      {/* No Faces Detected */}
      {results.faces && results.faces.length === 0 && (
        <motion.div
          className="text-center py-16"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
        >
          <motion.div
            className="glass rounded-3xl p-12 max-w-2xl mx-auto"
            animate={{ 
              boxShadow: [
                "0 0 20px rgba(255, 165, 0, 0.3)",
                "0 0 40px rgba(255, 165, 0, 0.5)",
                "0 0 20px rgba(255, 165, 0, 0.3)"
              ]
            }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            <div className="text-6xl mb-6">🔍</div>
            <h3 className="text-2xl font-bold text-yellow-400 mb-4">No Faces Detected</h3>
            <p className="text-gray-300 mb-8">
              The AI couldn't detect any faces in this image. Try using a clearer image with better lighting and visible faces.
            </p>
            <motion.button
              onClick={onReset}
              className="px-8 py-4 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-xl font-semibold text-white flex items-center gap-3 mx-auto hover:shadow-lg hover:shadow-yellow-500/25 transition-all duration-300"
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
            >
              <RefreshCw size={20} />
              Try Another Image
            </motion.button>
          </motion.div>
        </motion.div>
      )}
    </motion.div>
  )
}

export default ResultDisplay