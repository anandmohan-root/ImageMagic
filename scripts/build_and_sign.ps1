# Build the app with PyInstaller and sign BgReplacer.exe with a local,
# self-generated code-signing certificate.
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_and_sign.ps1
#
# The certificate is created once in your personal store (Cert:\CurrentUser\My),
# with a non-exportable private key, and reused for every build. It is NOT added
# to any trusted root store, so Windows still reports the signature as "not
# trusted"; on this PC, Smart App Control nevertheless allowed the signed .exe
# while blocking the identical unsigned build (tested 2026-11-08).
# Signing is fully offline: no timestamp server is contacted.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$subject = 'CN=BgReplacer (local code signing)'
$exe = Join-Path $root 'dist\BgReplacer\BgReplacer.exe'

$cert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert |
    Where-Object { $_.Subject -eq $subject -and $_.NotAfter -gt (Get-Date) } |
    Sort-Object NotAfter -Descending | Select-Object -First 1
if (-not $cert) {
    $cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject $subject `
        -CertStoreLocation Cert:\CurrentUser\My -KeyExportPolicy NonExportable `
        -NotAfter (Get-Date).AddYears(5)
    Write-Host "Created signing certificate $($cert.Thumbprint) (valid to $($cert.NotAfter.ToShortDateString()))"
} else {
    Write-Host "Using signing certificate $($cert.Thumbprint)"
}

Push-Location $root
try {
    # Run PyInstaller through the signed python.exe: Smart App Control can block
    # pip's unsigned pyinstaller.exe launcher.
    & (Join-Path $root '.venv\Scripts\python.exe') -m PyInstaller --noconfirm BgReplacer.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)" }
} finally { Pop-Location }

$null = Set-AuthenticodeSignature -FilePath $exe -Certificate $cert -HashAlgorithm SHA256
$sig = Get-AuthenticodeSignature $exe
if ($sig.SignerCertificate.Thumbprint -ne $cert.Thumbprint) { throw "Signing failed: $($sig.StatusMessage)" }
Write-Host "Signed $exe"
Write-Host "Signer: $($sig.SignerCertificate.Subject)  (self-signed: Windows shows it as not trusted)"
