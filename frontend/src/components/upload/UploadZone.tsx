import { useCallback, useRef, useState } from 'react'

import { VideoPreview } from './VideoPreview'

const ACCEPTED_TYPES = ['video/mp4', 'video/quicktime', 'video/webm']
const ACCEPTED_EXTENSIONS = '.mp4,.mov,.webm'
const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024

interface UploadZoneProps {
  readonly onSubmit: (file: File) => void
  readonly disabled?: boolean
  readonly progress?: number
}

function validateFile(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return 'Unsupported file type. Please upload an MP4, MOV, or WebM video.'
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return 'File is too large. Maximum size is 100 MB.'
  }
  return null
}

export function UploadZone({ onSubmit, disabled = false, progress }: UploadZoneProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback((file: File) => {
    const validationError = validateFile(file)
    if (validationError) {
      setError(validationError)
      setSelectedFile(null)
      return
    }
    setError(null)
    setSelectedFile(file)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) {
      handleFile(file)
    }
  }, [handleFile])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFile(file)
    }
  }, [handleFile])

  const handleBrowseClick = useCallback(() => {
    inputRef.current?.click()
  }, [])

  const handleClear = useCallback(() => {
    setSelectedFile(null)
    setError(null)
    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }, [])

  const handleSubmit = useCallback(() => {
    if (selectedFile && !disabled) {
      onSubmit(selectedFile)
    }
  }, [selectedFile, disabled, onSubmit])

  if (selectedFile && !disabled) {
    return (
      <VideoPreview
        file={selectedFile}
        onAnalyze={handleSubmit}
        onChange={handleClear}
      />
    )
  }

  return (
    <div className="upload-zone-wrapper">
      <div
        className={`upload-zone ${isDragOver ? 'upload-zone--drag-over' : ''} ${disabled ? 'upload-zone--disabled' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={disabled ? undefined : handleBrowseClick}
        role="button"
        tabIndex={disabled ? -1 : 0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            handleBrowseClick()
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS}
          onChange={handleInputChange}
          className="upload-zone__input"
          disabled={disabled}
        />
        <div className="upload-zone__icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>
        <p className="upload-zone__text">
          {disabled ? 'Uploading...' : 'Drag & drop your video here'}
        </p>
        {!disabled && (
          <p className="upload-zone__subtext">
            or <span className="upload-zone__link">browse files</span>
          </p>
        )}
        <p className="upload-zone__hint">MP4, MOV, or WebM &middot; Max 100 MB</p>
      </div>

      {disabled && progress !== undefined && (
        <div className="upload-zone__progress">
          <div className="progress-bar">
            <div className="progress-bar__fill" style={{ width: `${progress}%` }} />
          </div>
          <p className="upload-zone__progress-text">Uploading...</p>
        </div>
      )}

      {error && <p className="upload-zone__error">{error}</p>}
    </div>
  )
}
