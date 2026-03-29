export type AnalysisStatus = 'queued' | 'processing' | 'completed' | 'failed'

export type ProcessingStage =
  | 'validating'
  | 'extracting_events'
  | 'tribe_predict'
  | 'summarizing_rois'
  | 'generating_plots'
  | 'analyzing_with_claude'
  | 'saving_results'

export interface AnalysisCreateResponse {
  readonly analysis_id: string
  readonly status: AnalysisStatus
  readonly created_at: string
}

export interface AnalysisStatusResponse {
  readonly analysis_id: string
  readonly status: AnalysisStatus
  readonly stage?: ProcessingStage
  readonly stage_label?: string
  readonly progress_percent: number
}

export interface VideoMetadata {
  readonly filename: string
  readonly duration_seconds: number
  readonly resolution: string
  readonly file_size_bytes: number
}

export interface TimelineSegment {
  readonly start_seconds: number
  readonly end_seconds: number
  readonly label: string
  readonly dominant_regions: readonly string[]
  readonly cognitive_state: string
  readonly engagement_level: 'high' | 'medium' | 'low'
  readonly insight: string
}

export interface RegionInsight {
  readonly roi_name: string
  readonly full_name: string
  readonly cognitive_function: string
  readonly activation_rank: number
  readonly marketing_implication: string
}

export interface Recommendation {
  readonly category: string
  readonly title: string
  readonly description: string
  readonly priority: 'high' | 'medium' | 'low'
}

export interface AnalysisReport {
  readonly executive_summary: string
  readonly overall_score: number
  readonly timeline: readonly TimelineSegment[]
  readonly top_regions: readonly RegionInsight[]
  readonly recommendations: readonly Recommendation[]
  readonly strengths: readonly string[]
  readonly weaknesses: readonly string[]
}

export interface AnalysisResult {
  readonly analysis_id: string
  readonly status: AnalysisStatus
  readonly created_at: string
  readonly completed_at?: string
  readonly video_metadata: VideoMetadata
  readonly report: AnalysisReport
  readonly brain_images: readonly string[]
}

export interface ApiResponse<T> {
  readonly success: boolean
  readonly data?: T
  readonly error?: string
}
