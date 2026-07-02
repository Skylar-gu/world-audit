// Single source of copy: parses ../copy.md (invariant 7 — review copy there).
import raw from '../copy.md?raw'

const copy = {}
let key = null
for (const line of raw.split('\n')) {
  const m = line.match(/^## (.+)$/)
  if (m) { key = m[1].trim(); copy[key] = '' }
  else if (key && line.trim() && !line.startsWith('# ')) {
    copy[key] += (copy[key] ? ' ' : '') + line.trim()
  }
}
export default copy
