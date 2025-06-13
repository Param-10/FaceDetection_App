import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ImageUpload from './components/ImageUpload'
import ResultDisplay from './components/ResultDisplay'
import './App.css'

function App() {
  const [results, setResults] = useState(null)
  const [originalImage, setOriginalImage] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleImageUpload = async (file) => {
    setIsProcessing(true)
    setOriginalImage(URL.createObjectURL(file))
    
    try {
      const formData = new FormData()
      formData.append('image', file)
      
      const response = await fetch('/detect', {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      setResults(data)
    } catch (error) {
      console.error('Error processing image:', error)
      setResults({ error: error.message })
    } finally {
      setIsProcessing(false)
    }
  }

  const handleReset = () => {
    setResults(null)
    setOriginalImage(null)
    setIsProcessing(false)
  }

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 cyber-grid opacity-20"></div>
      
      {/* Floating Particles */}
      <div className="fixed inset-0 pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="particle"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -30, 0],
              opacity: [0.3, 1, 0.3],
            }}
            transition={{
              duration: 3 + Math.random() * 2,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <motion.header
          className="text-center mb-12"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <motion.h1
            className="text-6xl md:text-8xl font-bold gradient-text mb-4"
            animate={{
              backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
            }}
            transition={{
              duration: 5,
              repeat: Infinity,
              ease: "linear"
            }}
          >
            AI Face Detection
          </motion.h1>
          <motion.p
            className="text-xl text-gray-300 max-w-2xl mx-auto"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            Experience the future of facial recognition with advanced AI-powered analysis
          </motion.p>
        </motion.header>

        {/* Main Content */}
        <AnimatePresence mode="wait">
          {!results && !isProcessing && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.5 }}
            >
              <ImageUpload onImageUpload={handleImageUpload} />
            </motion.div>
          )}

          {isProcessing && (
            <motion.div
              key="processing"
              className="text-center py-20"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
            >
              <motion.div
                className="glass rounded-3xl p-12 max-w-2xl mx-auto"
                animate={{
                  boxShadow: [
                    "0 0 20px rgba(0, 245, 255, 0.3)",
                    "0 0 60px rgba(0, 245, 255, 0.6)",
                    "0 0 20px rgba(0, 245, 255, 0.3)"
                  ]
                }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <motion.div
                  className="w-24 h-24 mx-auto mb-8 rounded-full bg-gradient-to-r from-neon-blue to-neon-purple flex items-center justify-center"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                >
                  <div className="w-16 h-16 rounded-full border-4 border-white border-t-transparent animate-spin"></div>
                </motion.div>
                <h3 className="text-3xl font-bold gradient-text mb-4">
                  AI Processing
                </h3>
                <p className="text-gray-300 text-lg">
                  Analyzing facial features with advanced neural networks
                  <span className="loading-dots"></span>
                </p>
              </motion.div>
            </motion.div>
          )}

          {results && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -50 }}
              transition={{ duration: 0.6 }}
            >
              <ResultDisplay 
                results={results} 
                originalImage={originalImage}
                onReset={handleReset}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

export default App