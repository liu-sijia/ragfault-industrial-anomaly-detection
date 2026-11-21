$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath '.\.venv\Scripts\python.exe')) {
    throw '未找到 .venv。请先按 README 创建虚拟环境并安装 requirements.txt。'
}

& '.\.venv\Scripts\python.exe' '.\frontend\server.py'

