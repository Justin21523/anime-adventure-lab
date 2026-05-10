import type { WorldCharacterTemplate, WorldPack, WorldPlayerTemplate, WorldVisualStyle } from '../types/world.types'

export type WorldPackApplySelection = {
  world: boolean
  world_flags: boolean
  player_templates: boolean
  characters: boolean
  visual: boolean
}

export type WorldPackDiff = {
  worldChanged: boolean
  worldFields: Array<{ field: string; from: string; to: string }>
  worldFlagsChanged: Array<{ key: string; from: boolean | undefined; to: boolean | undefined }>
  visualChanged: boolean
  visualFields: Array<{ field: string; from: string; to: string }>
  charactersAdded: Array<{ id: string; name: string }>
  charactersUpdated: Array<{ id: string; name: string }>
  playerTemplatesAdded: Array<{ id: string; name: string }>
  playerTemplatesUpdated: Array<{ id: string; name: string }>
}

export function computeWorldPackDiff(base: WorldPack, candidate: WorldPack): WorldPackDiff {
  const worldFields: WorldPackDiff['worldFields'] = []
  const worldPairs: Array<[string, any, any]> = [
    ['name', base.name, candidate.name],
    ['description', base.description, candidate.description],
    ['setting', base.setting, candidate.setting],
    ['difficulty', base.difficulty, candidate.difficulty],
  ]
  for (const [field, from, to] of worldPairs) {
    if (String(from ?? '') !== String(to ?? '')) worldFields.push({ field, from: String(from ?? ''), to: String(to ?? '') })
  }

  const baseFlags = base.world_flags || {}
  const candFlags = candidate.world_flags || {}
  const keys = new Set<string>([...Object.keys(baseFlags), ...Object.keys(candFlags)])
  const worldFlagsChanged: WorldPackDiff['worldFlagsChanged'] = []
  for (const key of Array.from(keys).sort()) {
    const from = (baseFlags as any)[key]
    const to = (candFlags as any)[key]
    if (Boolean(from) !== Boolean(to)) worldFlagsChanged.push({ key, from: from as any, to: to as any })
  }

  const baseVisual = base.visual || ({} as any)
  const candVisual = candidate.visual || ({} as any)
  const visualFields: WorldPackDiff['visualFields'] = []
  const visualPairs: Array<[string, any, any]> = [
    ['prompt_prefix', baseVisual.prompt_prefix, candVisual.prompt_prefix],
    ['negative_prompt', baseVisual.negative_prompt, candVisual.negative_prompt],
    ['base_model', baseVisual.base_model, candVisual.base_model],
    ['default_loras', JSON.stringify(baseVisual.default_loras || []), JSON.stringify(candVisual.default_loras || [])],
  ]
  for (const [field, from, to] of visualPairs) {
    if (String(from ?? '') !== String(to ?? '')) visualFields.push({ field, from: String(from ?? ''), to: String(to ?? '') })
  }

  const baseChars = new Map((base.characters || []).map((c) => [c.character_id, c]))
  const candChars = new Map((candidate.characters || []).map((c) => [c.character_id, c]))
  const charactersAdded: WorldPackDiff['charactersAdded'] = []
  const charactersUpdated: WorldPackDiff['charactersUpdated'] = []
  for (const [id, c] of candChars.entries()) {
    if (!baseChars.has(id)) {
      charactersAdded.push({ id, name: c.name || id })
      continue
    }
    const before = baseChars.get(id)
    if (before && JSON.stringify(before) !== JSON.stringify(c)) {
      charactersUpdated.push({ id, name: c.name || id })
    }
  }

  const baseTpl = new Map((base.player_templates || []).map((t) => [t.template_id, t]))
  const candTpl = new Map((candidate.player_templates || []).map((t) => [t.template_id, t]))
  const playerTemplatesAdded: WorldPackDiff['playerTemplatesAdded'] = []
  const playerTemplatesUpdated: WorldPackDiff['playerTemplatesUpdated'] = []
  for (const [id, t] of candTpl.entries()) {
    if (!baseTpl.has(id)) {
      playerTemplatesAdded.push({ id, name: t.name || id })
      continue
    }
    const before = baseTpl.get(id)
    if (before && JSON.stringify(before) !== JSON.stringify(t)) {
      playerTemplatesUpdated.push({ id, name: t.name || id })
    }
  }

  return {
    worldChanged: worldFields.length > 0,
    worldFields,
    worldFlagsChanged,
    visualChanged: visualFields.length > 0,
    visualFields,
    charactersAdded,
    charactersUpdated,
    playerTemplatesAdded,
    playerTemplatesUpdated,
  }
}

export function applyWorldPackSelection(base: WorldPack, candidate: WorldPack, selection: WorldPackApplySelection): WorldPack {
  const next: WorldPack = JSON.parse(JSON.stringify(base)) as WorldPack

  if (selection.world) {
    next.name = candidate.name
    next.description = candidate.description
    next.setting = candidate.setting
    next.difficulty = candidate.difficulty
  }

  if (selection.world_flags) {
    next.world_flags = JSON.parse(JSON.stringify(candidate.world_flags || {})) as Record<string, boolean>
  }

  if (selection.visual) {
    next.visual = JSON.parse(JSON.stringify(candidate.visual)) as WorldVisualStyle
  }

  if (selection.player_templates) {
    const byId = new Map<string, WorldPlayerTemplate>()
    for (const t of next.player_templates || []) byId.set(t.template_id, t)
    for (const t of candidate.player_templates || []) byId.set(t.template_id, t)
    next.player_templates = Array.from(byId.values())
  }

  if (selection.characters) {
    const byId = new Map<string, WorldCharacterTemplate>()
    for (const c of next.characters || []) byId.set(c.character_id, c)
    for (const c of candidate.characters || []) byId.set(c.character_id, c)
    next.characters = Array.from(byId.values())
  }

  return next
}

