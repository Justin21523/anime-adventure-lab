import type { components } from './generated/api'
import apiClient, { apiGet, apiPost } from './client'

export type V2Capabilities = components['schemas']['V2CapabilitiesResponse']
export type V2World = components['schemas']['V2WorldResponse']
export type V2Session = components['schemas']['V2StorySessionResponse']
export type V2Turn = components['schemas']['V2StoryTurnResponse']
export type V2Job = components['schemas']['V2JobResponse']
export type V2JobEvent = components['schemas']['V2JobEventResponse']
export type V2DocumentUpload = components['schemas']['V2DocumentUploadResponse']
export type V2Document = components['schemas']['V2DocumentResponse']
export type V2ReviewProposal = components['schemas']['V2ReviewProposalResponse']
export type V2ReviewApproval = components['schemas']['V2ReviewApprovalResponse']
export type V2SystemStatus = components['schemas']['V2SystemStatusResponse']
export type V2Reconcile = components['schemas']['V2ReconcileResponse']

const baseURL = import.meta.env.VITE_API_V2_BASE || '/api/v2'

export function v2Get<T>(path: string): Promise<T> {
  return apiGet<T>(path, { baseURL, retry: false })
}

export function v2Post<T, D = undefined>(
  path: string,
  data?: D,
  headers?: Record<string, string>
): Promise<T> {
  return apiPost<T, D>(path, data, { baseURL, headers, retry: false })
}

export function v2Put<T, D>(
  path: string,
  data: D,
  headers?: Record<string, string>
): Promise<T> {
  return apiClient
    .put<T>(path, data, { baseURL, headers })
    .then((response) => response.data)
}

export async function uploadV2Document(
  worldId: string,
  file: File
): Promise<V2DocumentUpload> {
  const form = new FormData()
  form.append('file', file)
  const response = await apiClient.post<V2DocumentUpload>(
    `/worlds/${encodeURIComponent(worldId)}/documents`,
    form,
    {
      baseURL,
      headers: { 'Content-Type': 'multipart/form-data' },
    }
  )
  return response.data
}
