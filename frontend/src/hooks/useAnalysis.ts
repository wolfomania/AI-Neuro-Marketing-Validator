import { useCallback, useEffect, useRef, useState } from 'react'

import * as api from '../api/client'
import type { AnalysisResult } from '../types/analysis'

type AnalysisState =
  | { readonly phase: 'idle' }
  | { readonly phase: 'uploading'; readonly progress: number }
  | { readonly phase: 'processing'; readonly analysisId: string; readonly stage: string; readonly stageLabel: string; readonly progress: number }
  | { readonly phase: 'completed'; readonly analysisId: string; readonly result: AnalysisResult }
  | { readonly phase: 'error'; readonly message: string }

interface UseAnalysisReturn {
  readonly state: AnalysisState;
  readonly submitVideo: (file: File) => Promise<void>;
  readonly reset: () => void;
}

const POLL_INTERVAL_MS = 3000
const MAX_POLL_COUNT = 120

export function useAnalysis(): UseAnalysisReturn {
  const [state, setState] = useState<AnalysisState>({ phase: 'idle' })
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollCountRef = useRef(0)

  const clearPolling = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }, [])

  useEffect(() => {
    return clearPolling
  }, [clearPolling])

  const startPolling = useCallback((analysisId: string) => {
    clearPolling()
    pollCountRef.current = 0

    intervalRef.current = setInterval(async () => {
      pollCountRef.current++
      if (pollCountRef.current > MAX_POLL_COUNT) {
        clearPolling()
        setState({ phase: 'error', message: 'Analysis timed out. Please try again.' })
        return
      }

      const statusResponse = await api.getStatus(analysisId)

      if (!statusResponse.success || !statusResponse.data) {
        clearPolling()
        setState({ phase: 'error', message: statusResponse.error ?? 'Failed to fetch status' })
        return
      }

      const status = statusResponse.data

      if (status.status === 'completed') {
        clearPolling()
        const resultResponse = await api.getResult(analysisId)

        if (!resultResponse.success || !resultResponse.data) {
          setState({ phase: 'error', message: resultResponse.error ?? 'Failed to fetch results' })
          return
        }

        setState({ phase: 'completed', analysisId, result: resultResponse.data })
        return
      }

      if (status.status === 'failed') {
        clearPolling()
        setState({ phase: 'error', message: 'Analysis failed. Please try again with a different video.' })
        return
      }

      setState({
        phase: 'processing',
        analysisId,
        stage: status.stage ?? 'processing',
        stageLabel: status.stage_label ?? 'Processing...',
        progress: status.progress_percent,
      })
    }, POLL_INTERVAL_MS)
  }, [clearPolling])

  const submitVideo = useCallback(async (file: File) => {
    setState({ phase: 'uploading', progress: 0 })

    const response = await api.submitVideo(file)

    if (!response.success || !response.data) {
      setState({ phase: 'error', message: response.error ?? 'Failed to upload video' })
      return
    }

    const { analysis_id } = response.data

    setState({
      phase: 'processing',
      analysisId: analysis_id,
      stage: 'queued',
      stageLabel: 'Queued for processing...',
      progress: 0,
    })

    startPolling(analysis_id)
  }, [startPolling])

  const reset = useCallback(() => {
    clearPolling()
    setState({ phase: 'idle' })
  }, [clearPolling])

  return { state, submitVideo, reset }
}
