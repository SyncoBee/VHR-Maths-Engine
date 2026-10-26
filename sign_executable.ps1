<#
.SYNOPSIS
    VHR Maths Engine Pro - Digital Code Signing & Trust Setup Utility
    Created & Published by Vihanga Himantha

.DESCRIPTION
    Signs Windows binaries (VHR_Maths_Engine.exe and installer) with Authenticode signatures
    to eliminate 'Unknown Publisher' and Windows SmartScreen 'Don't run / Run with caution' warnings.

.EXAMPLE
    # Self-sign and trust on this PC (Development & Local testing)
    .\sign_executable.ps1 -Target "dist\VHR_Maths_Engine\VHR_Maths_Engine.exe" -CreateSelfSigned

.EXAMPLE
    # Sign with commercial PFX certificate (Production distribution)
    .\sign_executable.ps1 -Target "dist\VHR_Maths_Engine\VHR_Maths_Engine.exe" -PfxPath "C:\certs\mycert.pfx" -PfxPassword "mypassword"
#>

param (
    [string]$Target = "dist\VHR_Maths_Engine\VHR_Maths_Engine.exe",
    [switch]$CreateSelfSigned,
    [string]$PfxPath = "",
    [string]$PfxPassword = "",
    [string]$TimestampServer = "http://timestamp.digicert.com"
)

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " VHR Maths Engine Pro - Code Signing & Windows Trust Utility     " -ForegroundColor Cyan
Write-Host " Created & Published by Vihanga Himantha                         " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

if (-not (Test-Path $Target)) {
    Write-Warning "Target file '$Target' not found. Please build the executable first using 'python build_dist.py'."
    exit 1
}

$cert = $null

if ($PfxPath -ne "" -and (Test-Path $PfxPath)) {
    Write-Host "[*] Loading commercial PFX certificate from '$PfxPath'..." -ForegroundColor Green
    if ($PfxPassword -ne "") {
        $secPass = ConvertTo-SecureString $PfxPassword -AsPlainText -Force
        $cert = Get-PfxCertificate -FilePath $PfxPath
    } else {
        $cert = Get-PfxCertificate -FilePath $PfxPath
    }
}
elseif ($CreateSelfSigned) {
    Write-Host "[*] Generating a dedicated Authenticode Code Signing Certificate for Vihanga Himantha..." -ForegroundColor Yellow
    
    # Check if certificate already exists in CurrentUser\My
    $existing = Get-ChildItem -Path Cert:\CurrentUser\My -CodeSigningCert | Where-Object { $_.Subject -match "Vihanga Himantha" } | Select-Object -First 1
    
    if ($existing) {
        Write-Host "[+] Found existing Code Signing certificate for Vihanga Himantha." -ForegroundColor Green
        $cert = $existing
    } else {
        $cert = New-SelfSignedCertificate `
            -Type CodeSigning `
            -Subject "CN=Vihanga Himantha, O=VHR Maths Engine, OU=Software Publishing, C=LK" `
            -KeySpec Signature `
            -KeyLength 2048 `
            -KeyExportPolicy Exportable `
            -HashAlgorithm SHA256 `
            -CertStoreLocation "Cert:\CurrentUser\My" `
            -NotAfter (Get-Date).AddYears(5)
        Write-Host "[+] Successfully generated certificate: $($cert.Thumbprint)" -ForegroundColor Green
    }

    # Export public certificate for distribution/installation
    $outCertDir = Split-Path $Target -Parent
    $cerFile = Join-Path $outCertDir "Vihanga_Himantha_Public.cer"
    Export-Certificate -Cert $cert -FilePath $cerFile | Out-Null
    Write-Host "[+] Public certificate exported to: $cerFile" -ForegroundColor Cyan

    # Register to TrustedPublisher store (does not block with modal prompts)
    Write-Host "[*] Adding certificate to CurrentUser\TrustedPublisher store..." -ForegroundColor Yellow
    $pubStore = New-Object System.Security.Cryptography.X509Certificates.X509Store("TrustedPublisher", "CurrentUser")
    $pubStore.Open("ReadWrite")
    $pubStore.Add($cert)
    $pubStore.Close()
    Write-Host "[+] Certificate registered in CurrentUser\TrustedPublisher." -ForegroundColor Green
}
else {
    # Try finding any existing code signing cert
    $cert = Get-ChildItem -Path Cert:\CurrentUser\My -CodeSigningCert | Where-Object { $_.Subject -match "Vihanga Himantha" } | Select-Object -First 1
    if (-not $cert) {
        Write-Warning "No code signing certificate specified."
        Write-Host "Run with -CreateSelfSigned to generate a developer certificate, or pass -PfxPath for a commercial certificate." -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "[*] Signing '$Target' with SHA-256 and RFC 3161 Timestamping..." -ForegroundColor Green
try {
    $sig = Set-AuthenticodeSignature -FilePath $Target -Certificate $cert -TimestampServer $TimestampServer -HashAlgorithm SHA256
    if ($sig.Status -eq "Valid") {
        Write-Host "[+] SIGNATURE VALID! '$Target' is signed successfully." -ForegroundColor Green
        Write-Host "    Signer: $($sig.SignerCertificate.Subject)" -ForegroundColor White
        Write-Host "    Status: $($sig.StatusMessage)" -ForegroundColor White
    } else {
        Write-Host "[!] Signature status: $($sig.Status) ($($sig.StatusMessage))" -ForegroundColor Yellow
    }
} catch {
    Write-Error "Failed to sign binary: $_"
}
