function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

interface VideoPreviewProps {
  readonly file: File
  readonly onAnalyze: () => void
  readonly onChange: () => void
}

export function VideoPreview({ file, onAnalyze, onChange }: VideoPreviewProps) {
  return (
    <div className="video-preview">
      <div className="video-preview__icon">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="23 7 16 12 23 17 23 7" />
          <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
        </svg>
      </div>
      <div className="video-preview__info">
        <p className="video-preview__name">{file.name}</p>
        <p className="video-preview__size">{formatFileSize(file.size)}</p>
      </div>
      <div className="video-preview__actions">
        <button className="btn btn--primary" onClick={onAnalyze}>
          Analyze
        </button>
        <button className="btn btn--ghost" onClick={onChange}>
          Change
        </button>
      </div>
    </div>
  )
}
