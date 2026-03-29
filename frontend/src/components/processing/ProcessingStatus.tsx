interface ProcessingStatusProps {
  readonly stage: string
  readonly stageLabel: string
  readonly progress: number
}

export function ProcessingStatus({ stageLabel, progress }: ProcessingStatusProps) {
  return (
    <div className="processing">
      <div className="processing__spinner">
        <svg width="64" height="64" viewBox="0 0 64 64">
          <circle
            cx="32"
            cy="32"
            r="28"
            fill="none"
            stroke="var(--color-border)"
            strokeWidth="4"
          />
          <circle
            cx="32"
            cy="32"
            r="28"
            fill="none"
            stroke="var(--color-accent)"
            strokeWidth="4"
            strokeLinecap="round"
            strokeDasharray="176"
            strokeDashoffset={176 - (176 * progress) / 100}
            className="processing__spinner-ring"
          />
        </svg>
        <span className="processing__percent">{Math.round(progress)}%</span>
      </div>

      <p className="processing__label">{stageLabel}</p>

      <div className="progress-bar progress-bar--wide">
        <div
          className="progress-bar__fill"
          style={{ width: `${progress}%` }}
        />
      </div>

      <p className="processing__hint">This usually takes 1-3 minutes</p>
    </div>
  )
}
