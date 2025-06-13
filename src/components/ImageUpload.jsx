import React, { useRef, useState } from 'react'
import { motion } from 'framer-motion'

const ImageUpload = ({ onImageUpload }) => {
  const fileInputRef = useRef(null)
  const [dragActive, setDragActive] = useState(false)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file) => {
    if (file.type.startsWith('image/')) {
      onImageUpload(file)
    } else {
      alert('Please select an image file')
    }
  }

  const onButtonClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        className={`glass rounded-3xl p-12 border-2 border-dashed transition-all duration-300 ${
          dragActive 
            ? 'border-neon-blue bg-neon-blue/10' 
            : 'border-gray-600 hover:border-neon-blue/50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <div className="text-center">
          <motion.div
            className="w-32 h-32 mx-auto mb-8 rounded-full bg-gradient-to-r from-neon-blue to-neon-purple flex items-center justify-center"
            animate={{
              boxShadow: [
                "0 0 20px rgba(0, 245, 255, 0.3)",
                "0 0 40px rgba(0, 245, 255, 0.6)",
                "0 0 20px rgba(0, 245, 255, 0.3)"
              ]
            }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <svg
              className="w-16 h-16 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
          </motion.div>

          <h3 className="text-3xl font-bold gradient-text mb-4">
            Upload Your Image
          </h3>
          
          <p className="text-gray-300 text-lg mb-8">
            Drag & drop an image here, or click to select one
          </p>

          <motion.button
            onClick={onButtonClick}
            className="px-8 py-4 bg-gradient-to-r from-neon-blue to-neon-purple rounded-full text-white font-semibold text-lg hover:shadow-lg hover:shadow-neon-blue/25 transition-all duration-300"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Choose File
          </motion.button>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleChange}
            className="hidden"
          />

          <div className="mt-8 text-sm text-gray-400">
            <p>Supported formats: JPG, PNG, WEBP</p>
            <p>Maximum file size: 500KB for optimal performance</p>
          </div>
        </div>
      </motion.div>

      {/* Features Section */}
      <motion.div
        className="grid md:grid-cols-3 gap-6 mt-12"
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <div className="glass rounded-2xl p-6 text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-r from-neon-blue to-neon-purple flex items-center justify-center">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
          <h4 className="text-xl font-semibold text-white mb-2">Face Detection</h4>
          <p className="text-gray-400">Advanced AI algorithms to detect faces in your images</p>
        </div>

        <div className="glass rounded-2xl p-6 text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-r from-neon-purple to-neon-pink flex items-center justify-center">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h4 className="text-xl font-semibold text-white mb-2">Real-time Analysis</h4>
          <p className="text-gray-400">Fast processing with instant results</p>
        </div>

        <div className="glass rounded-2xl p-6 text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-r from-neon-pink to-neon-blue flex items-center justify-center">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </div>
          <h4 className="text-xl font-semibold text-white mb-2">Privacy First</h4>
          <p className="text-gray-400">Your images are processed securely and not stored</p>
        </div>
      </motion.div>
    </div>
  )
}

export default ImageUpload 