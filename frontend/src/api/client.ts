import type {
  AnalysisCreateResponse,
  AnalysisResult,
  AnalysisStatusResponse,
  ApiResponse,
} from '../types/analysis'

const API_BASE = '/api'

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem('nm_access_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function request<T>(url: string, options?: RequestInit): Promise<ApiResponse<T>> {
  try {
    const headers = { ...authHeaders(), ...options?.headers }
    const response = await fetch(`${API_BASE}${url}`, { ...options, headers })
    const body: unknown = await response.json()

    if (!response.ok) {
      const errorBody = body as { detail?: string }
      return {
        success: false,
        error: errorBody.detail ?? `Request failed with status ${response.status}`,
      }
    }

    return {
      success: true,
      data: body as T,
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Network error'
    return {
      success: false,
      error: message,
    }
  }
}

export async function submitVideo(
  file: File,
): Promise<ApiResponse<AnalysisCreateResponse>> {
  const formData = new FormData()
  formData.append('file', file)

  return request<AnalysisCreateResponse>('/analysis', {
    method: 'POST',
    body: formData,
  })
}

export async function getStatus(
  analysisId: string,
): Promise<ApiResponse<AnalysisStatusResponse>> {
  return request<AnalysisStatusResponse>(`/analysis/${analysisId}/status`)
}

export async function getResult(
  analysisId: string,
): Promise<ApiResponse<AnalysisResult>> {
  return request<AnalysisResult>(`/analysis/${analysisId}`)
}

export function getImageUrl(analysisId: string, imageName: string): string {
  return `${API_BASE}/analysis/${analysisId}/images/${imageName}`
}
