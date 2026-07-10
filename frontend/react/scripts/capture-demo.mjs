import { chromium } from 'playwright-core'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'

const baseURL = process.env.DEMO_BASE_URL || 'http://127.0.0.1:18081'
const password = process.env.DEMO_ADMIN_PASSWORD || ''
const executablePath = process.env.CHROMIUM_PATH || '/snap/bin/chromium'
const output = resolve(process.cwd(), '../../portfolio-web/assets')
const frames = mkdtempSync(join(tmpdir(), 'sagaforge-demo-'))
const captured = []

const browser = await chromium.launch({ executablePath, headless: true })
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 })
const page = await context.newPage()
await page.goto(baseURL, { waitUntil: 'networkidle' })
if (await page.getByRole('heading', { name: '登入 SagaForge' }).isVisible().catch(() => false)) {
  if (!password) throw new Error('DEMO_ADMIN_PASSWORD is required for the private demo')
  await page.getByLabel('密碼').fill(password)
  await page.getByRole('button', { name: '進入工作台' }).click()
  await page.getByRole('heading', { name: '故事工作區' }).waitFor()
}

async function capture(name) {
  const file = join(output, name)
  await page.screenshot({ path: file, fullPage: true })
  const frame = join(frames, `${String(captured.length).padStart(2, '0')}.png`)
  await page.screenshot({ path: frame, fullPage: false })
  captured.push(frame)
}

await capture('screenshot-dashboard.png')
await page.getByText('繼續故事 →').first().click()
await page.getByText('世界知識 · MinIO → pgvector').waitFor()
await capture('screenshot-knowledge.png')
await page.getByLabel('你的行動').fill('銀色檔案盒要如何安全開啟？')
await page.getByRole('button', { name: '送出行動' }).click()
await page.getByText(/工作執行中/).waitFor()
await capture('screenshot-job-running.png')
await page.getByText('RAG 引用證據').waitFor({ timeout: 30000 })
await capture('screenshot-story.png')
await page.getByText('RAG 引用證據').click()
await capture('screenshot-rag.png')
await page.getByText('Human-in-the-loop Review').scrollIntoViewIfNeeded()
await capture('screenshot-review.png')
await page.getByRole('button', { name: 'Approve' }).click()
await page.getByText('approved').waitFor()
await capture('screenshot-review-approved.png')
await page.getByText('Runtime & Services').scrollIntoViewIfNeeded()
await capture('screenshot-system.png')

const mobile = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 })
const mobilePage = await mobile.newPage()
await mobilePage.goto(baseURL, { waitUntil: 'networkidle' })
if (await mobilePage.getByRole('heading', { name: '登入 SagaForge' }).isVisible().catch(() => false)) {
  await mobilePage.getByLabel('密碼').fill(password)
  await mobilePage.getByRole('button', { name: '進入工作台' }).click()
  await mobilePage.getByRole('heading', { name: '故事工作區' }).waitFor()
}
await mobilePage.screenshot({ path: join(output, 'screenshot-mobile.png'), fullPage: true })
await mobilePage.screenshot({ path: join(frames, `${String(captured.length).padStart(2, '0')}.png`), fullPage: false })
captured.push(join(frames, `${String(captured.length).padStart(2, '0')}.png`))
await mobile.close()
await browser.close()

const concatFile = join(frames, 'frames.txt')
writeFileSync(concatFile, captured.map((file) => `file '${file}'\nduration 8`).join('\n') + `\nfile '${captured.at(-1)}'\n`)
const ffmpeg = spawnSync('ffmpeg', ['-y', '-f', 'concat', '-safe', '0', '-i', concatFile, '-vf', 'scale=1440:1000:force_original_aspect_ratio=decrease,pad=1440:1000:(ow-iw)/2:(oh-ih)/2,format=yuv420p', '-r', '30', '-c:v', 'libx264', '-movflags', '+faststart', join(output, 'demo-recording.mp4')], { stdio: 'inherit' })
if (ffmpeg.status !== 0) throw new Error('ffmpeg failed to produce the demo recording')
console.log(`Captured ${captured.length} real UI states in ${output}`)
