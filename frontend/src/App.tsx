import { ProcessingStatus } from './components/processing/ProcessingStatus'
import { ResultsView } from './components/results/ResultsView'
import { UploadZone } from './components/upload/UploadZone'
import { useAnalysis } from './hooks/useAnalysis'

function App() {
  const { state, submitVideo, reset } = useAnalysis()

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">Neuro-Marketing Validator</h1>
        <p className="app__subtitle">
          AI-powered brain activation analysis for video content
        </p>
      </header>

      <main className="app__main">
        {state.phase === 'idle' && (
          <UploadZone onSubmit={submitVideo} />
        )}

        {state.phase === 'uploading' && (
          <UploadZone
            onSubmit={submitVideo}
            disabled
            progress={state.progress}
          />
        )}

        {state.phase === 'processing' && (
          <ProcessingStatus
            stage={state.stage}
            stageLabel={state.stageLabel}
            progress={state.progress}
          />
        )}

        {state.phase === 'completed' && (
          <ResultsView result={state.result} onReset={reset} />
        )}

        {state.phase === 'error' && (
          <div className="error-card">
            <div className="error-card__icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--color-low)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
            </div>
            <h2 className="error-card__title">Something went wrong</h2>
            <p className="error-card__message">{state.message}</p>
            <button className="btn btn--primary" onClick={reset}>
              Try Again
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
